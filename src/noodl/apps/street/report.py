"""Units and output for the street application.

The model works in kg/m3 throughout (spec section 4); people read ug/m3, and AQ_DT's own
stage-3 product is a classic-CDF file with one record per (time, edge). Both live here so
that no unit conversion is written twice and no output format is invented twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

Tensor = torch.Tensor
UG_PER_KG = 1.0e9


def to_ug_m3(x: Tensor) -> Tensor:
    """kg/m3 -> ug/m3."""
    return torch.as_tensor(x, dtype=torch.float64) * UG_PER_KG


def from_ug_m3(x: Tensor) -> Tensor:
    """ug/m3 -> kg/m3."""
    return torch.as_tensor(x, dtype=torch.float64) / UG_PER_KG


def write_network_concentration(
    path,
    *,
    time_hours: Tensor,
    feature_index: Sequence[int],
    osmid: Sequence[int],
    background: Tensor,
    canyon_velocity: Tensor,
    concentration: Tensor,
    year: int | None = None,
    solver_background: float = 0.0,
) -> Path:
    """Write a `network_concentration_<year>.nc` in AQ_DT's own shape, as classic CDF.

    Dimensions `time` and `edge`; variables `time_hours`, `edge_feature_index`,
    `edge_osmid`, `forcing_background_concentration (time)`,
    `canyon_velocity_mps (time, edge)` and `concentration_increment (time, edge)` -- the
    names read off the real `leiden_small` product on 17 September 2026. Two deliberate
    differences from AQ_DT's own writer, both recorded as attributes on the file: this one
    writes CLASSIC CDF (AQ_DT writes NETCDF4, which `scipy.io.netcdf_file` cannot read at
    all) and float64 rather than float32.
    """
    from scipy.io import netcdf_file

    path = Path(path)
    concentration = torch.as_tensor(concentration, dtype=torch.float64)
    canyon_velocity = torch.as_tensor(canyon_velocity, dtype=torch.float64)
    time_hours = torch.as_tensor(time_hours, dtype=torch.float64)
    background = torch.as_tensor(background, dtype=torch.float64)
    n_time, n_edge = concentration.shape[-2], concentration.shape[-1]
    for name, value, expected in (
        ("canyon_velocity", canyon_velocity.shape[-2:], (n_time, n_edge)),
        ("time_hours", (time_hours.shape[-1],), (n_time,)),
        ("background", (background.shape[-1],), (n_time,)),
    ):
        if tuple(value) != tuple(expected):
            raise ValueError(
                f"write_network_concentration: {name} has trailing shape {tuple(value)}, "
                f"expected {tuple(expected)} to match concentration's "
                f"({n_time}, {n_edge})"
            )
    if len(feature_index) != n_edge or len(osmid) != n_edge:
        raise ValueError(
            f"write_network_concentration: feature_index has {len(feature_index)} "
            f"entries and osmid {len(osmid)}, but concentration has {n_edge} edges"
        )
    with netcdf_file(str(path), "w") as handle:
        handle.product_description = (
            "Network-only AQ product: canyon concentration increments on "
            "network-transport edges, written by noodl.apps.street.report"
        )
        handle.solver_background = str(float(solver_background))
        handle.file_format_note = (
            "classic CDF and float64, where AQ_DT's own writer uses NETCDF4 and float32"
        )
        if year is not None:
            handle.year = str(int(year))
        handle.createDimension("time", n_time)
        handle.createDimension("edge", n_edge)
        variable = handle.createVariable("time_hours", "d", ("time",))
        variable[:] = time_hours.numpy()
        variable = handle.createVariable("edge_feature_index", "i", ("edge",))
        variable[:] = torch.tensor(list(feature_index), dtype=torch.int32).numpy()
        variable = handle.createVariable("edge_osmid", "d", ("edge",))
        # OSM ids exceed 2**31, and classic CDF has no 64-bit integer; float64 holds them
        # exactly up to 2**53, which is four orders of magnitude above the largest in use.
        variable[:] = torch.tensor(list(osmid), dtype=torch.float64).numpy()
        variable = handle.createVariable(
            "forcing_background_concentration", "d", ("time",)
        )
        variable[:] = background.numpy()
        variable.units = "kg m-3"
        variable = handle.createVariable("canyon_velocity_mps", "d", ("time", "edge"))
        variable[:] = canyon_velocity.numpy()
        variable.units = "m s-1"
        variable = handle.createVariable("concentration_increment", "d",
                                         ("time", "edge"))
        variable[:] = concentration.numpy()
        variable.units = "kg m-3"
    return path
