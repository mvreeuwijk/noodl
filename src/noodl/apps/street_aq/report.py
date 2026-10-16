"""Units and output for the street application.

The model works in kg/m3 throughout; people read ug/m3, and the network product is a
classic-CDF file with one record per (time, edge). Both live here so that no unit
conversion is written twice and no output format is invented twice.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
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
    """Write a `network_concentration_<year>.nc` of per-street results, as classic CDF.

    Dimensions `time` and `edge`; variables `time_hours`, `edge_feature_index`,
    `edge_osmid`, `forcing_background_concentration (time)`,
    `canyon_velocity_mps (time, edge)` and `concentration_increment (time, edge)`. The file
    is CLASSIC CDF, which `scipy.io.netcdf_file` reads and writes (it cannot read NETCDF4),
    in float64 throughout; both choices are recorded as an attribute on the file.
    """
    from scipy.io import netcdf_file

    path = Path(path)

    def host(value, dtype=torch.float64) -> np.ndarray:
        """A detached, contiguous CPU float64 copy: valid for tensors that require grad."""
        return torch.as_tensor(value).detach().to("cpu", dtype).contiguous().numpy()

    concentration_np = host(concentration)
    canyon_np = host(canyon_velocity)
    time_np = host(time_hours)
    background_np = host(background)
    feature_np = np.asarray(list(feature_index), dtype=np.int32)
    osmid_np = np.asarray(list(osmid), dtype=np.float64)
    concentration = torch.from_numpy(concentration_np)
    canyon_velocity = torch.from_numpy(canyon_np)
    time_hours = torch.from_numpy(time_np)
    background = torch.from_numpy(background_np)
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
    # Write beside the destination and replace it only once the file is complete, so a
    # failure never truncates an existing product.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        _write_netcdf(netcdf_file, tmp_name, dict(
            n_time=n_time, n_edge=n_edge, year=year, solver_background=solver_background,
            time_np=time_np, feature_np=feature_np, osmid_np=osmid_np,
            background_np=background_np, canyon_np=canyon_np,
            concentration_np=concentration_np))
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def _write_netcdf(netcdf_file, target, v) -> None:
    n_time, n_edge = v["n_time"], v["n_edge"]
    year, solver_background = v["year"], v["solver_background"]
    with netcdf_file(target, "w") as handle:
        handle.product_description = (
            "Network-only AQ product: canyon concentration increments on "
            "network-transport edges, written by noodl.apps.street_aq.report"
        )
        handle.solver_background = str(float(solver_background))
        handle.file_format_note = (
            "classic CDF (readable by scipy.io.netcdf_file), float64 throughout"
        )
        if year is not None:
            handle.year = str(int(year))
        handle.createDimension("time", n_time)
        handle.createDimension("edge", n_edge)
        variable = handle.createVariable("time_hours", "d", ("time",))
        variable[:] = v["time_np"]
        variable = handle.createVariable("edge_feature_index", "i", ("edge",))
        variable[:] = v["feature_np"]
        variable = handle.createVariable("edge_osmid", "d", ("edge",))
        # OSM ids exceed 2**31, and classic CDF has no 64-bit integer; float64 holds them
        # exactly up to 2**53, which is four orders of magnitude above the largest in use.
        variable[:] = v["osmid_np"]
        variable = handle.createVariable(
            "forcing_background_concentration", "d", ("time",)
        )
        variable[:] = v["background_np"]
        variable.units = "kg m-3"
        variable = handle.createVariable("canyon_velocity_mps", "d", ("time", "edge"))
        variable[:] = v["canyon_np"]
        variable.units = "m s-1"
        variable = handle.createVariable("concentration_increment", "d",
                                         ("time", "edge"))
        variable[:] = v["concentration_np"]
        variable.units = "kg m-3"
