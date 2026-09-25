"""Hand-written AQ_DT products, in the exact formats of the real ones.

The GeoJSON property names, the NetCDF variable names, their dimensions and their dtypes
are copied from the AQ_DT repository's documented products and were checked against the real
`leiden_small` products. The fixture is WRITTEN rather than committed
as bytes: a classic-CDF file is not reviewable in a diff, whereas this module is, and
`scipy.io.netcdf_file` is the same reader the loader uses.

The network is four canyon streets and one `gaussian_fallback` service road:

    1004 ---(4)--- 1001 ---(0)--- 1002 ---(3)--- 1005
      |                             |
     (2, gaussian_fallback)        (1)
      |                             |
    1003 -------------------------- 1003
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NODES = {
    1001: (4.5300, 52.1600),
    1002: (4.5310, 52.1600),
    1003: (4.5310, 52.1610),
    1004: (4.5300, 52.1610),
    1005: (4.5320, 52.1600),
}

EDGES = [
    # (osmid, u, v, solver type, W_m, H_m, roughness_m or None)
    (900001, 1001, 1002, "network_transport", 20.0, 12.0, None),
    (900002, 1002, 1003, "network_transport", 15.0, 10.0, None),
    (900003, 1003, 1004, "gaussian_fallback", 30.0, 4.0, None),
    (900004, 1002, 1005, "network_transport", 18.0, 9.0, 0.2),
    (900005, 1004, 1001, "network_transport", 22.0, 11.0, None),
]

TIME_HOURS = [0.0, 3.0, 6.0, 9.0]
WIND_SPEED_MPS = [2.0, 5.0, 0.5, 8.0]
WIND_ANGLE_RAD = [0.0, 1.5707963267948966, 3.0, -2.0]
ABL_HEIGHT_M = [1200.0, 800.0, 300.0, 1500.0]
REFERENCE_HEIGHT_M = 30.0
BACKGROUND_CONCENTRATION = [1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9]
EMISSION_NORMALIZED = [0.10, 0.20, 0.30, 0.25, 0.15]
EMISSION_TIME_MODULATION = [0.5, 1.0, 1.5, 1.0]
SHUFFLE = [0, 3, 1, 4, 2]
"""The order the emission products are written in when `aligned=False`: features 0, 3, 1,
4, 2 -- the real `leiden_small` snapshot is in this state, its geometry file having been
regenerated after its emission products were written."""


def _node_feature(node_id: int, lon: float, lat: float) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "node_id": node_id,
            "degree": 2,
            "boundary": False,
            "cluster_size": 1,
            "significant_dead_end": False,
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def _edge_feature(osmid, u, v, solver, width, height, roughness) -> dict:
    properties = {
        "osmid": osmid,
        "osmids": [osmid],
        "highway": "residential",
        "source_highways": ["residential"],
        "functional_class": "residential",
        "junction": None,
        "is_roundabout": False,
        "name": f"Street {osmid}",
        "oneway": None,
        "lanes": None,
        "maxspeed": "30",
        "car_relevant": True,
        "repaired": True,
        "original_length_m": 100.0,
        "rewired_length_m": 100.0,
        "max_endpoint_shift_m": 0.0,
        "midpoint_offset_m": 0.0,
        "crosses_building": False,
        "u": u,
        "v": v,
        "significant_dead_end": False,
        "street_width_m": width,
        "W_m": width,
        "building_height_left_m": height,
        "building_height_right_m": height,
        "building_height_mean_m": height,
        "H_m": height,
        "height_width_ratio_mean": round(height / width, 3),
        "left_porosity": 0.0,
        "right_porosity": 0.0,
        "is_canyon": solver == "network_transport",
        "aq_solver_type": solver,
        "transect_sample_count": 2,
        "transect_success_count": 2,
        "geometry_confidence": 1.0,
    }
    if roughness is not None:
        properties["roughness_m"] = roughness
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "LineString",
            "coordinates": [list(NODES[u]), list(NODES[v])],
        },
    }


def _emission_feature(index: int) -> dict:
    osmid, u, v, _solver, _w, _h, _r = EDGES[index]
    return {
        "type": "Feature",
        "properties": {
            "osmid": osmid,
            "u": u,
            "v": v,
            "highway": "residential",
            "functional_class": "residential",
            "car_relevant": True,
            "emission_weight": 1.0,
            "emission_fraction": EMISSION_NORMALIZED[index],
            "emission_rate_nox_normalized": EMISSION_NORMALIZED[index],
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [list(NODES[u]), list(NODES[v])],
        },
    }


def _write_json(path: Path, features: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, indent=1)


def build(root, *, year: int = 2024, aligned: bool = True,
          kg_per_year: bool = True, reference_height_m=None,
          osmid_override: dict[int, object] | None = None) -> tuple[Path, Path]:
    """Write the five products under `root` and return `(stage1_dir, stage2_dir)`.

    `aligned=True` writes the emission products in the geometry file's own order, which is
    what a freshly built AQ_DT domain looks like. `aligned=False` writes them in `SHUFFLE`
    order while leaving `edge_index` at `0..n-1`, which is the state the real
    `leiden_small` products are in.

    `kg_per_year=False` writes `edge_emission_rate_nox_kg_per_year` as NaN, which is what
    the real `leiden_small` file contains at every one of its 904 rows.

    `reference_height_m` overrides the forcing's `reference_height_m` series; a sequence of
    `len(TIME_HOURS)` values writes a NON-constant label, which the loader must refuse. The
    default writes the real products' constant 30.0 m.

    `osmid_override` maps a feature index to the `osmid` value written for it in BOTH the
    geometry file and the emission key file, so that the emission key still matches. It is
    how a test writes an osmid the real products never carry (a string, say).
    """
    from scipy.io import netcdf_file

    root = Path(root)
    stage1 = root / "stage1_geometry" / "fixture"
    stage2 = root / "stage2_inputs" / "fixture"
    stage1.mkdir(parents=True, exist_ok=True)
    stage2.mkdir(parents=True, exist_ok=True)
    _write_json(stage1 / "repaired_nodes.geojson",
                [_node_feature(i, *lonlat) for i, lonlat in NODES.items()])
    overrides = dict(osmid_override or {})

    def _patch(index: int, feature: dict) -> dict:
        if index in overrides:
            feature["properties"]["osmid"] = overrides[index]
        return feature

    _write_json(stage1 / "repaired_edges_canyon.geojson",
                [_patch(i, _edge_feature(*edge)) for i, edge in enumerate(EDGES)])
    order = list(range(len(EDGES))) if aligned else list(SHUFFLE)
    _write_json(stage2 / "edge_emissions_normalized.geojson",
                [_patch(i, _emission_feature(i)) for i in order])

    n_time, n_edge = len(TIME_HOURS), len(EDGES)
    heights = ([REFERENCE_HEIGHT_M] * n_time if reference_height_m is None
               else list(reference_height_m))
    with netcdf_file(str(stage2 / f"forcing_{year}.nc"), "w") as handle:
        handle.history = "hand-written noodl fixture"
        handle.model_time_step_hours = "3"
        handle.createDimension("time", n_time)
        for name, values, units in (
            ("time_hours", TIME_HOURS, f"hours since {year}-01-01 00:00:00 UTC"),
            ("background_concentration", BACKGROUND_CONCENTRATION, None),
            ("wind_speed_mps", WIND_SPEED_MPS, None),
            ("wind_angle_rad", WIND_ANGLE_RAD, None),
            ("abl_height_m", ABL_HEIGHT_M, None),
            ("reference_height_m", heights, None),
        ):
            variable = handle.createVariable(name, "d", ("time",))
            variable[:] = np.asarray(values, dtype="float64")
            if units is not None:
                variable.units = units

    normalized = np.asarray([EMISSION_NORMALIZED[i] for i in order], dtype="float64")
    modulation = np.asarray(EMISSION_TIME_MODULATION, dtype="float64")
    with netcdf_file(str(stage2 / f"input_parameters_{year}.nc"), "w") as handle:
        handle.history = "hand-written noodl fixture"
        handle.emissions_policy = "Unit-source emissions only."
        handle.createDimension("time", n_time)
        handle.createDimension("edge", n_edge)
        variable = handle.createVariable("edge_index", "i", ("edge",))
        variable[:] = np.arange(n_edge, dtype="int32")
        variable = handle.createVariable("time_hours", "d", ("time",))
        variable[:] = np.asarray(TIME_HOURS, dtype="float64")
        variable.units = f"hours since {year}-01-01 00:00:00 UTC"
        variable = handle.createVariable("emission_time_modulation", "d", ("time",))
        variable[:] = modulation
        variable.units = "1"
        variable = handle.createVariable("edge_emission_weight", "d", ("edge",))
        variable[:] = np.ones(n_edge, dtype="float64")
        variable.units = "relative_weight"
        variable = handle.createVariable("edge_emission_fraction", "d", ("edge",))
        variable[:] = normalized
        variable.units = "1"
        variable = handle.createVariable(
            "edge_emission_rate_nox_normalized", "d", ("edge",)
        )
        variable[:] = normalized
        variable.units = "1"
        variable = handle.createVariable(
            "edge_emission_rate_nox_kg_per_year", "d", ("edge",)
        )
        variable[:] = (normalized * 1000.0 if kg_per_year
                       else np.full(n_edge, np.nan, dtype="float64"))
        variable.units = "kg yr-1"
        variable = handle.createVariable(
            "edge_emission_rate_nox_normalized_time", "d", ("time", "edge")
        )
        variable[:] = modulation[:, None] * normalized[None, :]
        variable.units = "1"
    return stage1, stage2
