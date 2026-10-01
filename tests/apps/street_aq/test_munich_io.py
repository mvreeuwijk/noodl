"""Street cases: the format-neutral `StreetCase`, and MUNICH's files read into and written
from it."""
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from noodl.apps.street_aq import build_model, munich_idealised
from noodl.apps.street_aq._munich_files import (
    deg_from_to_munich_rad,
    munich_rad_to_deg_from,
)
from noodl.apps.street_aq.case import StreetCase, drivers_at, read_case, write_case
from noodl.apps.street_aq.network import StreetNetwork

DATA = Path(__file__).resolve().parents[2] / "data" / "street"
# `munich_paris_excerpt` is four streets of the Le Perreux-sur-Marne network (MUNICH's
# published test case, east of Paris) with real per-street binaries; `munich_case_excerpt`
# is the same four streets with authored constant fields and one binary. See their NOTICEs.
MUNICH = dict(preset="munich", canyon_wind_min=0.0, u_d_min=0.0)
START = datetime(2014, 3, 16)


def _case(net, n_hours, *, wind_dir_from_deg=200.0, species=("NO2",), meteo_junction=None,
          **meteo):
    base = dict(wind_dir_from_deg=wind_dir_from_deg, wind_speed=5.0, h_abl=1000.0,
                u_star=0.5, lmo=1e6)
    base.update(meteo)
    return StreetCase.synthetic(
        net, species=species, times=[3600.0 * i for i in range(n_hours)], meteo=base,
        emissions=5e-6, background=2e-8, meteo_junction=meteo_junction, start=START,
    )


def _set_data_cfg(root, section, key, value):
    """Rewrites `key`'s line in `munich-data.cfg`'s `[section]`."""
    path = root / "munich-data.cfg"
    out, current = [], None
    for line in path.read_text().splitlines():
        if line.startswith("["):
            current = line.strip("[]")
        elif current == section and line.split(":")[0].split()[0:1] == [key]:
            line = f"{key}: {value}" if ":" in line else f"{key} {value}"
        out.append(line)
    path.write_text("\n".join(out) + "\n")


# ------------------------------------------------------------------------- conventions

def test_direction_is_toward_in_munich_and_from_in_noodl():
    # preprocessing/meteo.py:386 "0 for the wind to north": a north wind (FROM 0) blows
    # TOWARD 180, i.e. pi radians in MUNICH's file.
    assert munich_rad_to_deg_from(np.array([0.0]))[0] == pytest.approx(180.0)
    assert deg_from_to_munich_rad(np.array([0.0]))[0] == pytest.approx(math.pi)


# ------------------------------------------------------------------------------ reading

def test_excerpt_reads_per_street_arrays():
    case = read_case(DATA / "munich_paris_excerpt")
    n = len(case.street_ids)
    assert case.meteo["u_star"].shape == (3, n) and case.source == "munich"
    assert case.background.shape == (3, n, 1)
    assert len(case.junction_ids) == len(case.network.junctions)
    assert case.start == datetime(2014, 3, 16, 0)


def test_mixed_constant_and_binary_fields():
    # munich_case_excerpt: WindDirection is a binary, every other field an is_num constant.
    case = read_case(DATA / "munich_case_excerpt")
    assert case.meteo["wind_dir_from_deg"].shape == (2, 4)
    np.testing.assert_allclose(case.meteo["wind_dir_from_deg"][0], 180.0)     # toward N
    np.testing.assert_allclose(case.meteo["wind_dir_from_deg"][1], 270.0, atol=1e-5)
    np.testing.assert_array_equal(case.meteo["wind_speed"], np.full((2, 4), 5.0))
    np.testing.assert_array_equal(case.meteo["u_star"], np.full((2, 4), 0.4327))
    np.testing.assert_allclose(case.emissions, 1.3888888888888889e-05 / 1e9, rtol=1e-15)
    np.testing.assert_allclose(case.background, 20.0 / 1e9, rtol=1e-15)


def test_more_meteo_rows_than_nt_keeps_the_first_nt_rows(tmp_path):
    net, _ = munich_idealised()
    n = len(net.streets)
    write_case(tmp_path, _case(net, 2, wind_speed=np.array([5.0, 6.0])))
    # The written file holds Nt + 2 look-ahead rows; make every row distinct.
    rows = np.repeat(np.arange(10.0, 14.0)[:, None], n, axis=1)
    rows.astype("<f4").tofile(tmp_path / "meteo_WindSpeed.bin")
    case = read_case(tmp_path)
    assert len(case.times) == 2
    np.testing.assert_array_equal(case.meteo["wind_speed"], rows[:2])
    np.testing.assert_array_equal(case.meteo["h_abl"], np.full((2, n), 1000.0))   # constant


def test_a_data_section_starting_earlier_is_read_from_the_domain_start(tmp_path):
    net, _ = munich_idealised()
    n = len(net.streets)
    write_case(tmp_path, _case(net, 2, wind_speed=np.array([5.0, 6.0])))
    rows = np.repeat(np.arange(10.0, 14.0)[:, None], n, axis=1)
    rows.astype("<f4").tofile(tmp_path / "meteo_WindSpeed.bin")
    _set_data_cfg(tmp_path, "meteo", "Date_min", "2014-03-15-23")     # one step earlier
    case = read_case(tmp_path)
    np.testing.assert_array_equal(case.meteo["wind_speed"], rows[1:3])


def test_a_data_section_on_another_time_grid_is_refused(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 2, wind_speed=np.array([5.0, 6.0])))
    _set_data_cfg(tmp_path, "meteo", "Delta_t", "1800.0")
    with pytest.raises(ValueError, match=r"\[meteo\].*'WindSpeed'.*Delta_t 1800.0"):
        read_case(tmp_path)
    _set_data_cfg(tmp_path, "meteo", "Delta_t", "3600.0")
    _set_data_cfg(tmp_path, "meteo", "Date_min", "2014-03-16-01")     # starts too late
    with pytest.raises(ValueError, match="at or before the domain's start"):
        read_case(tmp_path)


def test_a_short_data_section_is_refused_not_cycled(tmp_path):
    net, _ = munich_idealised()
    n, n_j = len(net.streets), len(net.junctions)
    write_case(tmp_path, _case(net, 3, wind_speed=np.array([5.0, 6.0, 7.0])))
    np.full((2, n), 5.0, dtype="<f4").tofile(tmp_path / "meteo_WindSpeed.bin")
    _set_data_cfg(tmp_path, "meteo", "Nt", "2")
    with pytest.raises(ValueError, match=r"\[meteo\] field 'WindSpeed' has 2 records.*needs 3"):
        read_case(tmp_path)
    # A one-record field IS broadcast, like a constant. `WindSpeedInter` -- derived and
    # persisted by `write_case` since this case has no `meteo_junction` of its own -- shares
    # the same [meteo] section Nt, so it is re-written to match too.
    np.full((1, n), 4.0, dtype="<f4").tofile(tmp_path / "meteo_WindSpeed.bin")
    np.full((1, n_j), 4.0, dtype="<f4").tofile(tmp_path / "meteo_WindSpeedInter.bin")
    _set_data_cfg(tmp_path, "meteo", "Nt", "1")
    np.testing.assert_array_equal(read_case(tmp_path).meteo["wind_speed"],
                                  np.full((3, n), 4.0))


def test_malformed_files_are_named(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1))
    lines = (tmp_path / "munich-data.cfg").read_text().splitlines()
    head = lines.index("[background_concentration]")         # drop [emission]'s Filename
    lines[:head] = [ln for ln in lines[:head]                 # and its NO2 override
                    if not ln.startswith(("Filename", "NO2 "))]
    (tmp_path / "munich-data.cfg").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=r"\[emission\] has no 'Filename'"):
        read_case(tmp_path)
    write_case(tmp_path, _case(net, 1))
    street = (tmp_path / "street.dat").read_text().splitlines()
    street[1] += ";extra"
    (tmp_path / "street.dat").write_text("\n".join(street) + "\n")
    with pytest.raises(ValueError, match=r"street.dat row .* has 8 columns"):
        read_case(tmp_path)


# ------------------------------------------------------------------------------ writing

def test_round_trip_through_the_writer(tmp_path):
    net, _ = munich_idealised()
    series = np.repeat([10.0, 250.0], 2)                         # degrees FROM
    write_case(tmp_path, _case(net, 4, wind_dir_from_deg=series))
    case = read_case(tmp_path)
    np.testing.assert_allclose(case.meteo["wind_dir_from_deg"][:, 0], series,
                               atol=1e-4)                         # float32 files
    np.testing.assert_allclose(case.emissions[..., 0], 5e-6, rtol=1e-6)
    np.testing.assert_allclose(case.background[..., 0], 2e-8, rtol=1e-6)
    assert case.start == START and case.times == [0.0, 3600.0, 7200.0, 10800.0]


def test_a_read_case_writes_back_to_the_same_case(tmp_path):
    case = read_case(DATA / "munich_paris_excerpt")
    again = read_case(write_case(tmp_path, case))
    assert again.start == case.start and again.times == case.times
    assert again.street_ids == case.street_ids and again.species == case.species
    assert again.model_options() == case.model_options()
    for name in case.network.junctions:
        assert again.network.x[name] == pytest.approx(case.network.x[name], abs=1e-3)
        assert again.network.y[name] == pytest.approx(case.network.y[name], abs=1e-3)
    # Street directions come from the lon/lat themselves, MUNICH's rule, and the writer
    # writes those lon/lat back unchanged, so the directions survive the round trip.
    assert case.network.street_direction == "midlatitude_bearing"
    assert again.network.street_direction == "midlatitude_bearing"
    assert again.network.lon == case.network.lon and again.network.lat == case.network.lat
    assert again.network.azimuth == case.network.azimuth
    for key, value in case.meteo.items():
        np.testing.assert_allclose(again.meteo[key], value, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(again.emissions, case.emissions, rtol=1e-6)
    np.testing.assert_allclose(again.background, case.background, rtol=1e-6)


def test_writer_always_lists_the_six_mandatory_meteo_fields(tmp_path):
    # MUNICH v2.2 stops with "undefined variable SurfacePressure" without these, even with
    # chemistry, deposition and scavenging off (StreetNetworkTransport.cxx:750-776's
    # unconditional InitData calls, plus StreetNetworkChemistry.cxx:744-751's Attenuation).
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1))
    text = (tmp_path / "munich-data.cfg").read_text()
    meteo_section = text.split("[meteo]", 1)[1]
    fields_line = next(ln for ln in meteo_section.splitlines() if ln.startswith("Fields:"))
    expected = {
        "Rain": 0.0, "SolarRadiation": 0.0, "SpecificHumidity": 0.01,
        "SurfacePressure": 101325.0, "SurfaceTemperature": 293.15, "Attenuation": 1.0,
    }
    for name in expected:
        assert name in fields_line.split()
        line = next(ln for ln in meteo_section.splitlines() if ln.startswith(f"{name} "))
        assert float(line.split()[1]) == pytest.approx(expected[name])


def test_options_override_a_mandatory_meteo_field(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1), options={"SurfacePressure": 100000.0})
    data_text = (tmp_path / "munich-data.cfg").read_text()
    meteo_section = data_text.split("[meteo]", 1)[1]
    line = next(ln for ln in meteo_section.splitlines() if ln.startswith("SurfacePressure "))
    assert float(line.split()[1]) == pytest.approx(100000.0)
    # The override is a [meteo] value, not a [street] closure option.
    assert "SurfacePressure" not in (tmp_path / "munich.cfg").read_text()


def test_case_with_temperature_writes_its_own_surface_temperature(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1, temperature=310.0))
    meteo_section = (tmp_path / "munich-data.cfg").read_text().split("[meteo]", 1)[1]
    line = next(ln for ln in meteo_section.splitlines() if ln.startswith("SurfaceTemperature "))
    assert float(line.split()[1]) == pytest.approx(310.0)


def test_options_override_wins_over_a_value_the_case_already_supplies(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1, temperature=310.0),
               options={"SurfaceTemperature": 250.0})
    meteo_section = (tmp_path / "munich-data.cfg").read_text().split("[meteo]", 1)[1]
    line = next(ln for ln in meteo_section.splitlines() if ln.startswith("SurfaceTemperature "))
    assert float(line.split()[1]) == pytest.approx(250.0)


def test_a_non_numeric_meteo_override_is_named(tmp_path):
    net, _ = munich_idealised()
    with pytest.raises(ValueError, match=r"write_case: options\['SurfacePressure'\] = "
                                        r"'abc' is not a number"):
        write_case(tmp_path, _case(net, 1), options={"SurfacePressure": "abc"})


def test_missing_meteo_junction_is_derived_from_the_streets_touching_it(tmp_path):
    # MUNICH requires all five `...Inter` junction fields whenever `With_transport: yes`
    # (this writer's default) -- StreetNetworkTransport.cxx:1690-1699's "is needed but no
    # input data file was provided" checks -- even when the case carries no genuine
    # per-junction meteorology of its own.
    net, _ = munich_idealised()
    n = len(net.streets)
    col = {s.name: i for i, s in enumerate(net.streets)}
    # Junction "A" is touched by streets 1, 3, 4, 6; alternate their direction just either
    # side of north so the circular mean is ~0/360, where an arithmetic mean of angles in
    # [0, 360) would land near 180.
    wind_dir = np.full(n, 200.0)
    for name, value in (("1", 359.0), ("3", 1.0), ("4", 359.0), ("6", 1.0)):
        wind_dir[col[name]] = value
    # Junction "A"'s streets also get mixed-sign Obukhov lengths, so the derived `LMOInter`
    # can only match `1/mean(1/L)`: a plain arithmetic mean of these four (28.75, positive)
    # would fail the assertion below, which is negative.
    lmo = np.full(n, 1e6)
    for name, value in (("1", 50.0), ("3", -25.0), ("4", 100.0), ("6", -10.0)):
        lmo[col[name]] = value
    case = StreetCase.synthetic(
        net, species=("NO2",), times=[0.0],
        meteo=dict(wind_dir_from_deg=wind_dir[None, :], wind_speed=5.0, h_abl=1000.0,
                  u_star=0.5, lmo=lmo[None, :]),
        emissions=5e-6, background=2e-8, start=START,
    )
    write_case(tmp_path, case)
    meteo_section = (tmp_path / "munich-data.cfg").read_text().split("[meteo]", 1)[1]
    fields_line = next(ln for ln in meteo_section.splitlines() if ln.startswith("Fields:"))
    for name in ("WindDirectionInter", "WindSpeedInter", "PBLHInter", "USTInter", "LMOInter"):
        assert name in fields_line.split()
    again = read_case(tmp_path)
    # Junction ids are renumbered on write (they are not MUNICH's numeric form), but their
    # ORDER is preserved, so "A"'s position in the original network locates it in `again`.
    j = net.junctions.index("A")
    derived = again.meteo_junction["wind_dir_from_deg"][0, j] % 360.0
    assert min(derived, 360.0 - derived) < 0.1
    assert again.meteo_junction["wind_speed"][0, j] == pytest.approx(5.0)
    assert again.meteo_junction["h_abl"][0, j] == pytest.approx(1000.0)
    assert again.meteo_junction["u_star"][0, j] == pytest.approx(0.5)
    expected_lmo = 1.0 / np.mean(1.0 / np.array([50.0, -25.0, 100.0, -10.0]))
    assert expected_lmo < 0.0
    assert again.meteo_junction["lmo"][0, j] == pytest.approx(expected_lmo)


def test_a_supplied_meteo_junction_key_is_not_overwritten_by_the_derivation(tmp_path):
    net, _ = munich_idealised()
    n_j = len(net.junctions)
    supplied = np.arange(n_j, dtype=float)
    write_case(tmp_path, _case(net, 1, meteo_junction=dict(wind_speed=supplied[None, :])))
    case = read_case(tmp_path)
    np.testing.assert_allclose(case.meteo_junction["wind_speed"][0], supplied)
    # The other four are still derived (not left missing).
    for key in ("wind_dir_from_deg", "h_abl", "u_star", "lmo"):
        assert key in case.meteo_junction


def test_write_case_refuses_an_unknown_format(tmp_path):
    net, _ = munich_idealised()
    with pytest.raises(NotImplementedError, match="netcdf"):
        write_case(tmp_path, _case(net, 1), format="netcdf")


def test_write_case_refuses_missing_transport_meteo(tmp_path):
    # MUNICH needs PBLH, UST, LMO (and their *Inter counterparts) whenever `With_transport:
    # yes`, which this writer always sets -- a case missing any of the street-level fields
    # they come from must be refused up front, not written as a case MUNICH itself refuses.
    net, _ = munich_idealised()
    case = StreetCase.synthetic(
        net, species=("NO2",), times=[0.0],
        meteo=dict(wind_dir_from_deg=200.0, wind_speed=5.0, u_star=0.5),
        emissions=5e-6, background=2e-8, start=START,
    )
    with pytest.raises(ValueError, match=r"format='munich' needs meteo keys.*"
                                        r"missing \['h_abl', 'lmo'\]"):
        write_case(tmp_path, case)


def test_write_case_needs_a_start_date(tmp_path):
    net, _ = munich_idealised()
    case = StreetCase.synthetic(net, species=("NO2",), times=[0.0],
                                meteo=dict(wind_dir_from_deg=0.0, wind_speed=5.0),
                                emissions=0.0, background=0.0)
    with pytest.raises(ValueError, match="case.start"):
        write_case(tmp_path, case)


def test_synthetic_case_names_bad_inputs():
    net, _ = munich_idealised()
    kw = dict(species=("NO2",), times=[0.0, 3600.0], emissions=0.0, background=0.0)
    with pytest.raises(ValueError, match="missing.*wind_speed"):
        StreetCase.synthetic(net, meteo=dict(wind_dir_from_deg=0.0), **kw)
    with pytest.raises(ValueError, match="WindSpeed"):
        StreetCase.synthetic(net, meteo=dict(wind_dir_from_deg=0.0, wind_speed=5.0,
                                             WindSpeed=5.0), **kw)
    with pytest.raises(ValueError, match=r"meteo\['wind_speed'\].*\(2, 12\)"):
        StreetCase.synthetic(net, meteo=dict(wind_dir_from_deg=0.0,
                                             wind_speed=np.ones(3)), **kw)


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
    write_case(tmp_path, _case(sub, 2, meteo_junction=dict(wind_speed=inter)))
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
    write_case(tmp_path, _case(net, 1, meteo_junction=dict(wind_speed=ids[None, :])))

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


# ------------------------------------------------------------------------ model options

def test_model_options_reads_the_case_own_closure_settings():
    case = read_case(DATA / "munich_paris_excerpt")
    assert case.model_options() == dict(
        preset="munich", canyon_wind="exponential_profile",
        roof_exchange="aspect_ratio_scaled", roof_wind="bessel_canyon_mean",
        direction_averaging="rectangle_rule", z_ref=30.0, canyon_wind_min=0.1,
    )


def test_model_options_follow_non_default_keys(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1), options={
        "Mean_wind_speed_parameterization": "Sirane", "Transfer_parameterization": "Sirane",
        "Building_height_wind_speed_parameterization": "Macdonald",
        "With_horizontal_fluctuation": "no", "Zref": "20.0",
        "Minimum_Street_Wind_Speed": "0.25",
    })
    case = read_case(tmp_path)
    options = case.model_options()
    assert options == dict(
        preset="munich", canyon_wind="bessel_profile", roof_exchange="turbulent_velocity",
        roof_wind="canopy_log_law", direction_averaging="none", z_ref=20.0,
        canyon_wind_min=0.25,
    )
    build_model(case.network, species=("NO2",), **options)       # every keyword is valid


def test_model_options_defaults_a_missing_minimum_street_wind_speed(tmp_path):
    # MUNICH itself defaults `ustreet_min = 0.1` when the key is absent
    # (StreetNetworkTransport.cxx:164-168), not an error.
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1))
    lines = [ln for ln in (tmp_path / "munich.cfg").read_text().splitlines()
             if not ln.startswith("Minimum_Street_Wind_Speed")]
    (tmp_path / "munich.cfg").write_text("\n".join(lines) + "\n")
    case = read_case(tmp_path)
    assert case.model_options()["canyon_wind_min"] == pytest.approx(0.1)


def test_model_options_still_refuses_an_unparsable_minimum_street_wind_speed(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1), options={"Minimum_Street_Wind_Speed": "abc"})
    with pytest.raises(ValueError):
        read_case(tmp_path).model_options()


def test_model_options_refuse_what_noodl_does_not_implement(tmp_path):
    net, _ = munich_idealised()
    write_case(tmp_path, _case(net, 1), options={"Transfer_parameterization": "Wang"})
    with pytest.raises(NotImplementedError, match="Transfer_parameterization: Wang"):
        read_case(tmp_path).model_options()
    with pytest.raises(NotImplementedError, match="synthetic"):
        _case(net, 1).model_options()


def test_read_case_names_its_expectation_for_an_unknown_path(tmp_path):
    with pytest.raises(ValueError, match="munich.cfg"):
        read_case(tmp_path)


# ------------------------------------------------------------------------------ drivers

def test_drivers_follow_the_model_mode(tmp_path):
    net, _ = munich_idealised()
    rows = np.tile(np.linspace(10.0, 40.0, len(net.streets)), (3, 1))
    write_case(tmp_path, _case(net, 3, wind_dir_from_deg=rows))
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


def test_drivers_at_supplies_the_temperature_the_chemistry_reads(tmp_path):
    """`meteo["temperature"]` (K) becomes the `"temperature"` driver, at which the
    photostationary rate is evaluated; per street or as one network-wide mean."""
    net, _ = munich_idealised()
    # Whole kelvins, which MUNICH's single-precision files hold exactly.
    temps = np.tile(280.0 + 2.0 * np.arange(len(net.streets)), (2, 1))
    write_case(tmp_path, _case(net, 2, temperature=temps))
    case = read_case(tmp_path)
    per, _, _ = build_model(case.network, species=("NO2",), meteo="per_street", **MUNICH)
    d = drivers_at(case, per, 1)
    np.testing.assert_allclose(d["temperature"].numpy(), temps[1], rtol=1e-12)
    uni, _, _ = build_model(case.network, species=("NO2",), **MUNICH)
    assert float(drivers_at(case, uni, 1)["temperature"]) == pytest.approx(float(temps[1].mean()),
                                                                  rel=1e-12)
    assert "temperature" not in drivers_at(_case(net, 1), uni, 0)


def test_drivers_at_names_its_mismatches():
    net, _ = munich_idealised()
    case = _case(net, 1)
    model, _, _ = build_model(net, species=("NO2",), **MUNICH)
    with pytest.raises(ValueError, match="species \\['O3'\\]"):
        drivers_at(case, model, 0, species=("O3",))
    no_wind = StreetCase(**{**case.__dict__, "meteo": {"wind_dir_from_deg":
                                                       case.meteo["wind_dir_from_deg"]}})
    with pytest.raises(KeyError, match="wind_speed"):
        drivers_at(no_wind, model, 0)
    reordered = StreetNetwork(streets=list(reversed(net.streets)), x=net.x, y=net.y)
    other, _, _ = build_model(reordered, species=("NO2",), **MUNICH)
    with pytest.raises(ValueError, match="street order"):
        drivers_at(case, other, 0)


def test_circular_reduction_across_north(tmp_path):
    net, _ = munich_idealised()
    n = len(net.streets)
    toward = np.where(np.arange(n) % 2 == 0, 2 * math.pi - 0.05, 0.05)
    write_case(tmp_path, _case(net, 1, wind_dir_from_deg=munich_rad_to_deg_from(toward)[None]))
    case = read_case(tmp_path)
    uni, _, _ = build_model(case.network, species=("NO2",), **MUNICH)
    theta = float(drivers_at(case, uni, 0)["theta_w"])
    # toward ~north = FROM ~180; noodl theta_w is CCW from east TOWARD -> ~pi/2
    assert abs(math.remainder(theta - math.pi / 2, 2 * math.pi)) < 1e-3


def test_direction_exactly_on_the_wrap(tmp_path):
    # MUNICH directions of exactly 0 and exactly 2 pi -- the latter as a float32 file holds
    # it, 6.2831855 > 2 pi -- are the same direction: toward north, noodl theta_w = pi/2.
    net, _ = munich_idealised()
    n, n_j = len(net.streets), len(net.junctions)
    write_case(tmp_path, _case(net, 1, wind_dir_from_deg=np.linspace(0.0, 90.0, n)[None]))
    row = np.where(np.arange(n) % 2 == 0, 0.0, 2 * math.pi).astype("<f4")
    assert float(row[1]) > 2 * math.pi
    np.tile(row, (3, 1)).tofile(tmp_path / "meteo_WindDirection.bin")
    # `WindDirectionInter` was derived and persisted by `write_case` from the ORIGINAL
    # (pre-overwrite) linspace directions; with every street now exactly north, every
    # junction's circular mean is trivially also exactly north, so its binary is re-written
    # the same way (a differently-sized array of the same two representations).
    row_j = np.where(np.arange(n_j) % 2 == 0, 0.0, 2 * math.pi).astype("<f4")
    np.tile(row_j, (3, 1)).tofile(tmp_path / "meteo_WindDirectionInter.bin")
    case = read_case(tmp_path)
    np.testing.assert_allclose(case.meteo["wind_dir_from_deg"], 180.0, atol=1e-4)
    uni, _, _ = build_model(case.network, species=("NO2",), **MUNICH)
    theta = float(drivers_at(case, uni, 0)["theta_w"])
    assert abs(math.remainder(theta - math.pi / 2, 2 * math.pi)) < 1e-6
    per, _, _ = build_model(case.network, species=("NO2",), meteo="per_street", **MUNICH)
    junction = drivers_at(case, per, 0)["theta_w_junction"]
    assert max(abs(math.remainder(float(t) - math.pi / 2, 2 * math.pi)) for t in junction) \
        < 1e-6
