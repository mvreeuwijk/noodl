"""MUNICH case files: reading, writing and building drivers."""
import math
from pathlib import Path

import numpy as np
import pytest

from noodl.apps.street_aq import build_model, munich_idealised
from noodl.apps.street_aq.case import (
    deg_from_to_munich_rad,
    drivers_at,
    munich_rad_to_deg_from,
    read_case,
    write_case,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "street"
MUNICH = dict(canyon_wind="exponential", exchange="schulte", stability="munich",
              direction_averaging="munich", roof_wind_form="sirane")


def test_direction_is_toward_in_munich_and_from_in_noodl():
    # preprocessing/meteo.py:386 "0 for the wind to north": a north wind (FROM 0) blows
    # TOWARD 180, i.e. pi radians in MUNICH's file.
    assert munich_rad_to_deg_from(np.array([0.0]))[0] == pytest.approx(180.0)
    assert deg_from_to_munich_rad(np.array([0.0]))[0] == pytest.approx(math.pi)


def test_excerpt_reads_per_street_arrays():
    case = read_case(DATA / "munich_paris_excerpt")
    n = len(case.street_ids)
    assert case.meteo["u_star"].shape == (3, n) and case.source == "munich"
    assert case.background.shape == (3, n, 1)
    assert len(case.junction_ids) == len(case.network.junctions)


def test_round_trip_through_the_writer(tmp_path):
    net, _ = munich_idealised()
    series = np.repeat([0.3, 1.2], 2)                          # radians TOWARD
    write_case(tmp_path, net, format="munich", species=("NO2",), date_min="2014-03-16-00",
               n_hours=4, meteo=dict(WindDirection=series, WindSpeed=5.0, PBLH=1000.0,
                                     UST=0.5, LMO=1e6),
               emissions_kg_s=5e-6, background_kg_m3=2e-8)
    case = read_case(tmp_path)
    np.testing.assert_allclose(case.meteo["wind_dir_from_deg"][:, 0],
                               (np.degrees(series) + 180.0) % 360.0,
                               atol=1e-4)                       # float32 files
    np.testing.assert_allclose(case.emissions[..., 0], 5e-6, rtol=1e-6)
    np.testing.assert_allclose(case.background[..., 0], 2e-8, rtol=1e-6)


def test_drivers_follow_the_model_mode(tmp_path):
    net, _ = munich_idealised()
    rows = np.tile(np.linspace(0.1, 0.4, len(net.streets)), (3, 1))
    write_case(tmp_path, net, format="munich", species=("NO2",), date_min="2014-03-16-00",
               n_hours=3, meteo=dict(WindDirection=rows, WindSpeed=5.0, PBLH=1000.0, UST=0.5,
                                     LMO=1e6),
               emissions_kg_s=5e-6, background_kg_m3=2e-8)
    case = read_case(tmp_path)
    per, _, _ = build_model(case.network, species=("NO2",), meteo="per_street",
                            background="per_street", **MUNICH)
    d = drivers_at(case, per, 0)
    assert d["theta_w"].shape == (len(net.streets),)
    assert d["street.x_boundary"].shape == (len(net.streets),)
    assert "theta_w_junction" in d and float(d["u_star"][0]) == pytest.approx(0.5)
    uni, _, _ = build_model(case.network, species=("NO2",), **MUNICH)
    du = drivers_at(case, uni, 0)
    assert du["theta_w"].dim() == 0 and du["street.x_boundary"].shape == (1,)


def test_more_meteo_rows_than_nt_and_mixed_constant_fields(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, net, format="munich", species=("NO2",), date_min="2014-03-16-00",
               n_hours=2, meteo=dict(WindDirection=np.array([0.1, 0.2]), WindSpeed=5.0,
                                     PBLH=1000.0, UST=0.5, LMO=1e6),
               emissions_kg_s=5e-6, background_kg_m3=2e-8)
    case = read_case(tmp_path)
    assert len(case.times) == 2 and case.meteo["wind_speed"].shape == (2, len(net.streets))


def test_circular_reduction_across_north(tmp_path):
    net, _ = munich_idealised()
    n = len(net.streets)
    row = np.where(np.arange(n) % 2 == 0, 2 * math.pi - 0.05, 0.05)
    write_case(tmp_path, net, format="munich", species=("NO2",), date_min="2014-03-16-00",
               n_hours=1, meteo=dict(WindDirection=row[None, :], WindSpeed=5.0, PBLH=1000.0,
                                     UST=0.5, LMO=1e6),
               emissions_kg_s=5e-6, background_kg_m3=2e-8)
    case = read_case(tmp_path)
    uni, _, _ = build_model(case.network, species=("NO2",), **MUNICH)
    theta = float(drivers_at(case, uni, 0)["theta_w"])
    # toward ~north = FROM ~180; noodl theta_w is CCW from east TOWARD -> ~pi/2
    assert abs(math.remainder(theta - math.pi / 2, 2 * math.pi)) < 1e-3
