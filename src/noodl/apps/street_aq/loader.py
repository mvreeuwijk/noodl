"""Read the AQ_DT products into a `StreetNetwork`, a `Forcing` and an emission table.

The products are the ones `.superpowers/aqdt-repo-research.md` documents, read with the
standard library `json` and `scipy.io.netcdf_file` -- both inputs are classic CDF, checked.
Nothing here writes to the AQ_DT tree.

TWO INPUT AMBIGUITIES, both handled explicitly rather than inherited (spec section 6.3):

* **Units.** `background_concentration` is CAMS EAC4 NO2 MASS MIXING RATIO (kg/kg) written
  with no `units` attribute at all. It is converted here with `rho_air = 1.2041 kg/m3` and
  the assumption is recorded on the returned object. An unlabelled number never passes
  through.
* **Wind height.** `reference_height_m` reads 30.0 in the file, but the wind is ERA5's 10 m
  `u10`/`v10` with no extrapolation. The height is therefore an explicit ARGUMENT,
  defaulting to the physical truth of 10.0 m, and a file that disagrees raises unless
  `trust_file_height=True` accepts the disagreement with the file's `reference_height_m`
  label and proceeds with `wind_height_m` (the label itself is recorded in `notes`; the
  IMPAQ comparison proceeds with 30 m, because the prototype uses 30 m).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from noodl.apps.street_aq.canyon import Z0_B_DEFAULT
from noodl.apps.street_aq.network import Street, StreetNetwork

_DTYPE = torch.float64
EARTH_RADIUS_M = 6371000.0
RHO_AIR = 1.2041
"""kg/m3, the density the mass-mixing-ratio conversion assumes."""


@dataclass(frozen=True)
class Forcing:
    """One domain-wide forcing series. Every field is `(n_time,)` except the two scalars."""

    time_hours: torch.Tensor
    u_ref: torch.Tensor
    theta_w: torch.Tensor
    h_abl: torch.Tensor
    background: torch.Tensor
    wind_height_m: float
    reference_height_m: float


@dataclass(frozen=True)
class AqdtData:
    net: StreetNetwork
    forcing: Forcing
    emission: torch.Tensor
    feature_index: list[int]
    osmid: list[int]
    notes: dict[str, str]


def project(lon, lat, ref_lat: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Equirectangular metres about `ref_lat`, as `impaq_loader.project` computes them."""
    lon = torch.as_tensor(lon, dtype=_DTYPE)
    lat = torch.as_tensor(lat, dtype=_DTYPE)
    scale = EARTH_RADIUS_M * math.cos(math.radians(float(ref_lat)))
    return scale * torch.deg2rad(lon), EARTH_RADIUS_M * torch.deg2rad(lat)


def _load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first(properties: dict, keys: Sequence[str], default):
    for key in keys:
        value = properties.get(key)
        if value not in (None, ""):
            return value
    return default


def _feature_key(properties: dict) -> tuple:
    return (properties.get("osmid"), properties.get("u"), properties.get("v"))


def _netcdf(path: Path):
    from scipy.io import netcdf_file

    return netcdf_file(str(path), "r", mmap=False)  # closed by its own `with` block


def _column(handle, name: str, where: str) -> torch.Tensor:
    if name not in handle.variables:
        raise KeyError(
            f"read_aqdt: {where} has no variable {name!r}; it has "
            f"{sorted(handle.variables)}"
        )
    return torch.as_tensor(handle.variables[name].data.astype("float64"), dtype=_DTYPE)


def read_aqdt(
    stage1_dir,
    stage2_dir,
    *,
    year: int,
    select: str = "network_transport",
    emissions: str = "normalized",
    wind_height_m: float = 10.0,
    trust_file_height: bool = False,
    z0_b: float = Z0_B_DEFAULT,
    align: str = "emission_key",
    times: Sequence[int] | slice | None = None,
) -> AqdtData:
    """`repaired_nodes.geojson` + `repaired_edges_canyon.geojson` + the two NetCDFs.

    `select` picks the `aq_solver_type` to keep; features are taken by `is_canyon is True`
    instead when no feature carries `aq_solver_type` at all.

    `align` decides how the emission rows are matched to the selected features.
    `"emission_key"` (the default) matches by the `(osmid, u, v)` key of
    `edge_emissions_normalized.geojson`, whose feature order IS the NetCDF's row order.
    `"edge_index"` uses the `edge_index` variable as a position into the geometry file,
    which is the AQ_DT contract -- and is VERIFIED here rather than trusted, because on the
    `leiden_small` snapshot of 17 September 2026 it is wrong: the geometry file was
    regenerated (2026-06-09T17:17:50) after the parameters file was written
    (08:55:55), it gained a feature, and 515 of the 904 rows land on a different feature
    than their own. The verification is what turns that into an error instead of a
    plausible-looking wrong answer.

    `wind_height_m` is the height the forcing wind is taken to be valid at, and
    `trust_file_height=True` accepts a disagreement with the file's `reference_height_m`
    label and proceeds with `wind_height_m` anyway; the label itself is always recorded in
    `notes["wind_height"]`, whichever height the load used.

    `times` selects forcing steps (a slice or a sequence of indices); the default reads all
    of them, which is 2928 for a 2024 domain.
    """
    if emissions not in ("normalized", "kg_per_year"):
        raise ValueError(
            f"read_aqdt: emissions must be 'normalized' or 'kg_per_year', got "
            f"{emissions!r}"
        )
    if align not in ("emission_key", "edge_index"):
        raise ValueError(
            f"read_aqdt: align must be 'emission_key' or 'edge_index', got {align!r}"
        )
    stage1, stage2 = Path(stage1_dir), Path(stage2_dir)
    nodes = _load_json(stage1 / "repaired_nodes.geojson")["features"]
    if not nodes:
        raise ValueError(f"read_aqdt: {stage1 / 'repaired_nodes.geojson'} has no features")
    lons = [float(f["geometry"]["coordinates"][0]) for f in nodes]
    lats = [float(f["geometry"]["coordinates"][1]) for f in nodes]
    ref_lat = sum(lats) / len(lats)
    x_all, y_all = project(lons, lats, ref_lat)
    x = {str(int(f["properties"]["node_id"])): float(v)
         for f, v in zip(nodes, x_all, strict=True)}
    y = {str(int(f["properties"]["node_id"])): float(v)
         for f, v in zip(nodes, y_all, strict=True)}

    features = _load_json(stage1 / "repaired_edges_canyon.geojson")["features"]
    labelled = any("aq_solver_type" in f.get("properties", {}) for f in features)
    chosen: list[int] = []
    for index, feature in enumerate(features):
        properties = feature.get("properties", {})
        if labelled:
            if properties.get("aq_solver_type") != select:
                continue
        elif properties.get("is_canyon") is not True:
            continue
        chosen.append(index)
    rule = (f"aq_solver_type == {select!r}" if labelled
            else "is_canyon is True (no feature carries aq_solver_type)")
    if not chosen:
        raise ValueError(
            f"read_aqdt: no feature of {stage1 / 'repaired_edges_canyon.geojson'} has "
            f"{rule}"
        )
    streets: list[Street] = []
    osmid: list[int] = []
    for index in chosen:
        properties = features[index]["properties"]
        u, v = str(int(properties["u"])), str(int(properties["v"]))
        for node in (u, v):
            if node not in x:
                raise KeyError(
                    f"read_aqdt: feature {index} (osmid {properties.get('osmid')}) names "
                    f"node {node}, which is not in repaired_nodes.geojson"
                )
        length = math.hypot(x[v] - x[u], y[v] - y[u])
        streets.append(Street(
            name=str(index), u=u, v=v, length=length,
            width=float(_first(properties, ("W_m", "street_width_m"), 0.0)),
            height=float(_first(properties, ("H_m", "building_height_mean_m"), 0.0)),
            z0_b=float(_first(properties, ("roughness_m",), z0_b)),
        ))
        value = properties.get("osmid")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"read_aqdt: feature {index} of "
                f"{stage1 / 'repaired_edges_canyon.geojson'} has osmid {value!r}, which "
                f"is not an integer; both Leiden domains carry integer osmids everywhere, "
                f"so this is an unrecognised product rather than a value to stand in for"
            )
        osmid.append(int(value))
    net = StreetNetwork(streets=streets, x=x, y=y)

    forcing_path = stage2 / f"forcing_{year}.nc"
    where = str(forcing_path)
    with _netcdf(forcing_path) as handle:
        time_hours = _column(handle, "time_hours", where)
        u_ref = _column(handle, "wind_speed_mps", where)
        theta_w = _column(handle, "wind_angle_rad", where)
        h_abl = _column(handle, "abl_height_m", where)
        background = _column(handle, "background_concentration", where)
        file_height = _column(handle, "reference_height_m", where)
    reference_height = float(file_height.reshape(-1)[0])
    if float(file_height.max()) != float(file_height.min()):
        raise ValueError(
            f"read_aqdt: reference_height_m is not constant in {where} (it runs from "
            f"{float(file_height.min())} to {float(file_height.max())} m); this loader "
            f"assumes one wind height for the whole series"
        )
    if not trust_file_height and abs(reference_height - wind_height_m) > 1e-9:
        raise ValueError(
            f"read_aqdt: {where} labels its wind as valid at {reference_height} m, and "
            f"wind_height_m is {wind_height_m} m. On the AQ_DT products the label is "
            f"metadata and the wind is ERA5's 10 m u10/v10 with no extrapolation, so the "
            f"label is wrong; pass trust_file_height=True to proceed with the height "
            f"you gave; the file's label is recorded in `notes`. The IMPAQ comparison "
            f"proceeds with wind_height_m={reference_height} that way, because the "
            f"prototype uses the file's height"
        )
    if times is not None:
        selector = (torch.arange(len(time_hours))[times] if isinstance(times, slice)
                    else torch.as_tensor(list(times), dtype=torch.long))
        time_hours = time_hours[selector]
        u_ref = u_ref[selector]
        theta_w = theta_w[selector]
        h_abl = h_abl[selector]
        background = background[selector]
    forcing = Forcing(
        time_hours=time_hours, u_ref=u_ref, theta_w=theta_w, h_abl=h_abl,
        background=background * RHO_AIR, wind_height_m=float(wind_height_m),
        reference_height_m=reference_height,
    )

    parameters_path = stage2 / f"input_parameters_{year}.nc"
    where = str(parameters_path)
    with _netcdf(parameters_path) as handle:
        edge_index = handle.variables["edge_index"].data.astype("int64").tolist()
        rows = _column(handle, "edge_emission_rate_nox_normalized_time", where)
        if emissions == "kg_per_year":
            per_year = _column(handle, "edge_emission_rate_nox_kg_per_year", where)
            normalized = _column(handle, "edge_emission_rate_nox_normalized", where)
    if emissions == "kg_per_year":
        if not bool(torch.isfinite(per_year).all()):
            bad = int((~torch.isfinite(per_year)).sum())
            raise ValueError(
                f"read_aqdt: edge_emission_rate_nox_kg_per_year is not finite at {bad} of "
                f"{per_year.numel()} rows of {where} (it is entirely NaN on the "
                f"leiden_small snapshot of 17 September 2026); emissions='kg_per_year' "
                f"cannot be scaled from it -- use emissions='normalized'"
            )
        safe = torch.where(normalized > 0, normalized, torch.ones_like(normalized))
        scale = torch.where(normalized > 0, per_year / (safe * 365.25 * 86400.0),
                            torch.zeros_like(normalized))
        rows = rows * scale

    row_of_feature = _emission_rows(stage2, features, edge_index, align, where)
    missing = [i for i in chosen if i not in row_of_feature]
    if missing:
        raise ValueError(
            f"read_aqdt: {len(missing)} selected features have no emission row in "
            f"{where}; the first is feature {missing[0]} (osmid "
            f"{features[missing[0]]['properties'].get('osmid')})"
        )
    columns = torch.tensor([row_of_feature[i] for i in chosen], dtype=torch.long)
    emission = rows.index_select(-1, columns)
    if times is not None:
        emission = emission[selector]

    notes = {
        "background": (
            "CAMS EAC4 NO2 mass mixing ratio (kg/kg) in the file, with no units "
            f"attribute; converted to kg/m3 with rho_air = {RHO_AIR} kg/m3"
        ),
        "wind_height": (
            f"the file labels its wind valid at {reference_height} m; this load used "
            f"{wind_height_m} m (ERA5 u10/v10, no extrapolation)"
        ),
        "emissions": (
            "dimensionless unit source as AQ_DT writes it; the concentrations are a unit "
            "response" if emissions == "normalized" else
            "kg/s, from edge_emission_rate_nox_kg_per_year spread over the year"
        ),
        "alignment": f"emission rows matched to features by {align!r}",
        "selection": f"{rule}: {len(chosen)} of {len(features)}",
    }
    return AqdtData(net=net, forcing=forcing, emission=emission,
                    feature_index=list(chosen), osmid=osmid, notes=notes)


def _emission_rows(stage2: Path, features, edge_index, align: str, where: str
                   ) -> dict[int, int]:
    """Feature index -> emission row, verified against `edge_emissions_normalized.geojson`.

    That file's feature order IS the parameters NetCDF's row order (checked on
    `leiden_small`: `edge_emission_rate_nox_normalized[i]` equals feature `i`'s
    `emission_rate_nox_normalized` exactly, at every one of its 904 rows), and it carries
    the `(osmid, u, v)` key that identifies a feature independently of any position. A key
    that occurs more than once in that file is a contradiction -- it is never resolved by
    last-write-wins, because that would make an unselected feature silently donate its
    emission series to a selected one with the same key.
    """
    key_path = stage2 / "edge_emissions_normalized.geojson"
    by_key: dict[tuple, int] = {}
    if key_path.exists():
        for row, feature in enumerate(_load_json(key_path)["features"]):
            key = _feature_key(feature["properties"])
            if key in by_key:
                raise ValueError(
                    f"read_aqdt: emission key (osmid, u, v) = {key} occurs at features "
                    f"{by_key[key]} and {row} of {key_path}; alignment by emission key "
                    f"is ambiguous"
                )
            by_key[key] = row
    if align == "emission_key":
        if not by_key:
            raise FileNotFoundError(
                f"read_aqdt: align='emission_key' needs {key_path}, which is not there; "
                f"pass align='edge_index' to use the positional contract instead"
            )
        out: dict[int, int] = {}
        for index, feature in enumerate(features):
            row = by_key.get(_feature_key(feature["properties"]))
            if row is not None:
                out[index] = row
        return out
    positional = {int(feature): row for row, feature in enumerate(edge_index)}
    if by_key:
        wrong: list[int] = []
        for index, feature in enumerate(features):
            row = by_key.get(_feature_key(feature["properties"]))
            if row is not None and positional.get(index) != row:
                wrong.append(index)
        if wrong:
            shown = wrong[:5]
            raise ValueError(
                f"read_aqdt: align='edge_index' puts {len(wrong)} emission rows on a "
                f"feature whose (osmid, u, v) does not match (the first features: "
                f"{shown}), so the geometry file and {where} describe different edge "
                f"orders -- the products are out of step. Use align='emission_key', "
                f"which matches by that key instead"
            )
    return positional
