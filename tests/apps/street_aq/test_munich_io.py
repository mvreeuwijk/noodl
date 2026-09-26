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
from noodl.apps.street_aq.network import StreetNetwork

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


def test_model_options_reads_the_case_own_closure_settings():
    case = read_case(DATA / "munich_paris_excerpt")
    options = case.model_options()
    assert options["canyon_wind"] == "exponential"
    assert options["exchange"] == "schulte"
    assert options["stability"] == "munich"
    assert options["direction_averaging"] == "munich"
    assert options["roof_wind_form"] == "sirane"
    assert options["canyon_wind_min"] == pytest.approx(0.1)


def test_read_case_names_both_expectations_for_an_unknown_path(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        read_case(tmp_path)
    assert "MUNICH" in str(excinfo.value) and "SIRANE" in str(excinfo.value)


def test_write_case_refuses_a_non_munich_format(tmp_path):
    net, _ = munich_idealised()
    with pytest.raises(NotImplementedError, match="sirane"):
        write_case(tmp_path, net, format="sirane", species=("NO2",),
                   date_min="2014-03-16-00", n_hours=1,
                   meteo=dict(WindDirection=0.1, WindSpeed=5.0, PBLH=1000.0, UST=0.5,
                              LMO=1e6),
                   emissions_kg_s=5e-6, background_kg_m3=2e-8)


def test_inter_meteo_uses_the_junction_count_not_the_street_count(tmp_path):
    # A subnetwork whose street count and junction count DIFFER: writing an `...Inter`
    # array with the street count instead would either be rejected outright or (on the
    # unlucky subnetwork where n_streets < n_junctions) silently write too few columns.
    net, _ = munich_idealised()
    sub_streets = [s for s in net.streets if s.name in ("1", "4", "6")]
    sub = StreetNetwork(streets=sub_streets, x=net.x, y=net.y)
    n_streets, n_junctions = len(sub.streets), len(sub.junctions)
    assert n_streets != n_junctions
    inter = np.tile(np.arange(n_junctions, dtype=float), (2, 1))
    write_case(tmp_path, sub, format="munich", species=("NO2",), date_min="2014-03-16-00",
               n_hours=2, meteo=dict(WindSpeedInter=inter, WindSpeed=5.0, PBLH=1000.0,
                                     WindDirection=0.1, UST=0.5, LMO=1e6),
               emissions_kg_s=5e-6, background_kg_m3=2e-8)
    case = read_case(tmp_path)
    assert case.meteo_junction["wind_speed"].shape == (2, n_junctions)
    np.testing.assert_allclose(case.meteo_junction["wind_speed"][0], np.arange(n_junctions))


def test_meteo_junction_survives_a_permuted_intersection_file(tmp_path):
    # `intersection.dat`'s row order need not agree with `network.junctions` order (this
    # writer's own output happens to agree, but a real MUNICH file's need not): the reader
    # must look columns up BY ID, not by position. Column c holds junction c's own
    # (1-based) id, so a value surviving the permutation intact proves the lookup is by id.
    net, _ = munich_idealised()
    n = len(net.junctions)
    ids = np.arange(1, n + 1, dtype=float)
    write_case(tmp_path, net, format="munich", species=("NO2",), date_min="2014-03-16-00",
               n_hours=1, meteo=dict(WindSpeedInter=ids[None, :], WindSpeed=5.0,
                                     PBLH=1000.0, WindDirection=0.1, UST=0.5, LMO=1e6),
               emissions_kg_s=5e-6, background_kg_m3=2e-8)

    lines = (tmp_path / "intersection.dat").read_text().splitlines()
    header, rows = lines[0], lines[1:]
    assert len(rows) == n
    perm = list(reversed(range(n)))                     # a non-trivial reordering
    (tmp_path / "intersection.dat").write_text(
        "\n".join([header] + [rows[i] for i in perm]) + "\n"
    )
    raw = np.fromfile(tmp_path / "meteo_WindSpeedInter.bin", dtype="<f4").reshape(-1, n)
    raw[:, perm].astype("<f4").tofile(tmp_path / "meteo_WindSpeedInter.bin")

    case = read_case(tmp_path)
    np.testing.assert_allclose(case.meteo_junction["wind_speed"][0], ids)


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
