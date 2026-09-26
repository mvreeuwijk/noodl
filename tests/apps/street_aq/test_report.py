"""Units and the NetCDF product."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.street_aq.report import (
    from_ug_m3,
    to_ug_m3,
    write_network_concentration,
)

DT = torch.float64


def test_the_unit_helpers_are_exact_inverses():
    x = torch.tensor([1.0e-9, 5.5e-8, 0.0], dtype=DT)
    torch.testing.assert_close(to_ug_m3(x), x * 1e9, rtol=0, atol=0)
    torch.testing.assert_close(from_ug_m3(to_ug_m3(x)), x, rtol=1e-15, atol=0)
    # 40 ug/m3 is the EU annual NO2 limit value, a typical magnitude to sanity-check on.
    assert abs(float(from_ug_m3(torch.tensor(40.0, dtype=DT))) - 4.0e-8) < 1e-23


def test_the_product_round_trips_through_scipy(tmp_path):
    from scipy.io import netcdf_file

    n_time, n_edge = 4, 3
    concentration = torch.arange(n_time * n_edge, dtype=DT).reshape(n_time, n_edge) * 1e-9
    canyon = torch.linspace(-3.0, 3.0, n_time * n_edge, dtype=DT).reshape(n_time, n_edge)
    path = write_network_concentration(
        tmp_path / "network_concentration_2024.nc",
        time_hours=torch.tensor([0.0, 3.0, 6.0, 9.0], dtype=DT),
        feature_index=[6, 7, 26],
        osmid=[7434480, 52100759, 7435409],
        background=torch.tensor([1e-9, 2e-9, 3e-9, 4e-9], dtype=DT),
        canyon_velocity=canyon,
        concentration=concentration,
        year=2024,
    )
    handle = netcdf_file(str(path), "r", mmap=False)
    try:
        assert dict(handle.dimensions) == {"time": n_time, "edge": n_edge}
        assert set(handle.variables) == {
            "time_hours", "edge_feature_index", "edge_osmid",
            "forcing_background_concentration", "canyon_velocity_mps",
            "concentration_increment",
        }
        assert handle.variables["concentration_increment"].dimensions == ("time", "edge")
        # Classic CDF is big-endian on disk, which torch will not adopt directly, so
        # the round trip goes through `astype("float64")`.
        torch.testing.assert_close(
            torch.as_tensor(
                handle.variables["concentration_increment"].data.astype("float64"),
                dtype=DT,
            ), concentration, rtol=0, atol=0,
        )
        torch.testing.assert_close(
            torch.as_tensor(
                handle.variables["canyon_velocity_mps"].data.astype("float64"), dtype=DT
            ), canyon, rtol=0, atol=0,
        )
        # OSM ids exceed 2**31, so they are stored as float64 and must come back exact.
        assert [int(v) for v in handle.variables["edge_osmid"].data] == [
            7434480, 52100759, 7435409
        ]
        assert [int(v) for v in handle.variables["edge_feature_index"].data] == [6, 7, 26]
        assert handle.solver_background == b"0.0"
    finally:
        handle.close()


def test_a_shape_mismatch_is_named(tmp_path):
    concentration = torch.zeros(4, 3, dtype=DT)
    with pytest.raises(ValueError, match=r"canyon_velocity has trailing shape"):
        write_network_concentration(
            tmp_path / "out.nc", time_hours=torch.zeros(4, dtype=DT),
            feature_index=[0, 1, 2], osmid=[1, 2, 3],
            background=torch.zeros(4, dtype=DT),
            canyon_velocity=torch.zeros(4, 2, dtype=DT), concentration=concentration,
        )
    with pytest.raises(ValueError, match=r"feature_index has 2 entries"):
        write_network_concentration(
            tmp_path / "out.nc", time_hours=torch.zeros(4, dtype=DT),
            feature_index=[0, 1], osmid=[1, 2, 3],
            background=torch.zeros(4, dtype=DT),
            canyon_velocity=torch.zeros(4, 3, dtype=DT), concentration=concentration,
        )
