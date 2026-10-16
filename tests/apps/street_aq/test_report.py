"""Units and the NetCDF product."""

from __future__ import annotations

import os
import stat

import numpy as np
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


def _kwargs(concentration, tmp_path):
    n_time, n_edge = concentration.shape
    return dict(
        time_hours=torch.zeros(n_time, dtype=DT), feature_index=list(range(n_edge)),
        osmid=list(range(n_edge)), background=torch.zeros(n_time, dtype=DT),
        canyon_velocity=torch.zeros(n_time, n_edge, dtype=DT), concentration=concentration,
    )


def test_a_differentiable_solve_result_exports(tmp_path):
    from scipy.io import netcdf_file

    from noodl.apps.street_aq import build_model, munich_idealised

    net, _ = munich_idealised()
    model, state, _ = build_model(net, species=("NO2",), preset="munich",
                                  canyon_wind_min=0.0, u_d_min=0.0)
    from noodl.apps.street_aq.case import StreetCase, drivers_at

    scale = torch.tensor(1.0, dtype=DT, requires_grad=True)
    case = StreetCase.synthetic(
        net, species=("NO2",), times=[0.0], emissions=5e-6, background=2e-8,
        meteo=dict(wind_dir_from_deg=200.0, wind_speed=5.0, h_abl=1000.0, u_star=0.5,
                   lmo=1e6))
    d = drivers_at(case, model, 0)
    d["street.sources"] = d["street.sources"] * scale
    x = model.steady(state, d)["street.x"]
    assert x.requires_grad
    conc = x[None]
    path = write_network_concentration(
        tmp_path / "out.nc",
        time_hours=torch.zeros(1, dtype=DT), feature_index=list(range(conc.shape[-1])),
        osmid=list(range(conc.shape[-1])), background=torch.zeros(1, dtype=DT, requires_grad=True),
        canyon_velocity=torch.zeros(1, conc.shape[-1], dtype=DT, requires_grad=True),
        concentration=conc,
    )
    handle = netcdf_file(str(path), "r", mmap=False)
    try:
        got = handle.variables["concentration_increment"].data.astype("float64")
    finally:
        handle.close()
    np.testing.assert_array_equal(got, conc.detach().numpy())


def test_a_failed_export_leaves_an_existing_destination_untouched(tmp_path):
    dest = tmp_path / "out.nc"
    dest.write_bytes(b"previous product")
    bad = torch.zeros(2, 3, dtype=DT)
    kw = _kwargs(bad, tmp_path)
    kw["feature_index"] = [0, 1]          # wrong length: validation fails
    with pytest.raises(ValueError):
        write_network_concentration(dest, **kw)
    assert dest.read_bytes() == b"previous product"
    # A failure while writing (after the temp file exists) also keeps it, and cleans up.
    kw = _kwargs(bad, tmp_path)
    kw["osmid"] = ["x", "y", "z"]
    with pytest.raises((ValueError, TypeError)):
        write_network_concentration(dest, **kw)
    assert dest.read_bytes() == b"previous product"
    assert [p.name for p in tmp_path.iterdir()] == ["out.nc"]


def test_a_failure_after_the_temp_file_exists_cleans_up(tmp_path, monkeypatch):
    from noodl.apps.street_aq import report

    dest = tmp_path / "out.nc"
    dest.write_bytes(b"previous product")
    seen = []

    def boom(netcdf_file, target, values):
        seen.append(target)
        with open(target, "wb") as handle:    # the temp file exists, partly written
            handle.write(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(report, "_write_netcdf", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_network_concentration(dest, **_kwargs(torch.zeros(2, 3, dtype=DT), tmp_path))
    assert seen and seen[0] != str(dest)
    assert dest.read_bytes() == b"previous product"
    assert [p.name for p in tmp_path.iterdir()] == ["out.nc"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_product_has_the_umask_mode_not_mkstemps_0600(tmp_path):
    old = os.umask(0o027)
    try:
        dest = write_network_concentration(
            tmp_path / "out.nc", **_kwargs(torch.zeros(2, 3, dtype=DT), tmp_path))
    finally:
        os.umask(old)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o640
