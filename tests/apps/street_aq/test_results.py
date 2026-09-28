"""`read_results`: the format-neutral `StreetResults`, read from a SIRANE result directory
(`RUES_PAR_HEURE/`, `METEO/Resul_Meteo.dat`) or a MUNICH `results/` directory of
`<species>.bin` files."""
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from noodl.apps.street_aq import SIRANE_EXCHANGE, munich_idealised
from noodl.apps.street_aq.case import StreetCase, read_case, read_results

STREET_DATA = Path(__file__).resolve().parents[2] / "data" / "street"
DATA = STREET_DATA / "sirane_south_kensington"
MASTER = DATA / "Donnees_SouthKensington.dat"
RESULT_DIR = DATA / "RESULT_SOUTHKENSINGTON"
MUNICH_EXCERPT = STREET_DATA / "munich_paris_excerpt"

@pytest.fixture(scope="module")
def case() -> StreetCase:
    return read_case(MASTER)


def _write_stale_hour(src: Path, dst: Path, *, sigma_w: float) -> None:
    """A result file from an earlier, unrelated run: the deck's own hour-00 file with its
    `Sigma_wH` column replaced, so the hour is recognisably its own when read back."""
    lines = src.read_text().splitlines()
    col = lines[0].split("\t").index("Sigma_wH")
    out = [lines[0]]
    for line in lines[1:]:
        fields = line.split("\t")
        fields[col] = f"{sigma_w:.2f}"
        out.append("\t".join(fields))
    dst.write_text("\n".join(out) + "\n")


def _munich_case(species=("NO2",), n_hours=1, *, start=datetime(2014, 3, 16)):
    net, _ = munich_idealised()
    return net, StreetCase.synthetic(
        net, species=species, times=[3600.0 * i for i in range(n_hours)],
        meteo=dict(wind_dir_from_deg=200.0, wind_speed=5.0),
        emissions=0.0, background=0.0, start=start,
    )


# --------------------------------------------------------------------- SIRANE: reading

def test_hours_case_reads_only_the_deck_s_own_two_hours(case):
    results = read_results(RESULT_DIR, case=case)
    assert results.source == "sirane"
    assert results.times == [datetime(2014, 1, 7, 0), datetime(2014, 1, 7, 1)]
    assert len(results.street_ids) == 46 and results.street_ids == case.street_ids
    assert results.species == ["NO2", "NO", "O3"]
    for sp in results.species:
        assert results.c_in[sp].shape == (2, 46)
        assert results.c_above[sp].shape == (2, 46)
    assert results.u_canyon.shape == (2, 46)
    assert results.sigma_w_roof.shape == (2, 46)
    assert results.u_exchange.shape == (2, 46)


def test_hours_all_includes_the_stale_archived_hours(case, tmp_path):
    result_dir = tmp_path / "RESULT_SOUTHKENSINGTON"
    shutil.copytree(RESULT_DIR, result_dir)
    hourly = result_dir / "RUES_PAR_HEURE"
    for hour in ("12", "13"):
        _write_stale_hour(hourly / "Rues_2014010700.dat",
                          hourly / f"Rues_20140107{hour}.dat", sigma_w=1.06)

    only_the_deck = read_results(result_dir, case=case)
    assert only_the_deck.times == [datetime(2014, 1, 7, 0), datetime(2014, 1, 7, 1)]

    every_hour = read_results(result_dir, case=case, hours="all")
    assert every_hour.times == [
        datetime(2014, 1, 7, 0), datetime(2014, 1, 7, 1),
        datetime(2014, 1, 7, 12), datetime(2014, 1, 7, 13),
    ]
    assert every_hour.u_canyon.shape == (4, 46)
    # The stale hours carry their own (larger) Sigma_wH -- confirms they were actually
    # read, not just counted.
    np.testing.assert_allclose(every_hour.sigma_w_roof[2], 1.06)
    # Resul_Meteo.dat was overwritten by the later (coherent) run and has no row for the
    # stale hours 12/13 at all -- their meteo reads as nan rather than raising.
    assert np.isnan(every_hour.meteo["u_star"][2:]).all()
    assert not np.isnan(every_hour.meteo["u_star"][:2]).any()


def test_hours_all_needs_no_case():
    results = read_results(RESULT_DIR, hours="all")
    assert results.times == [datetime(2014, 1, 7, 0), datetime(2014, 1, 7, 1)]
    assert results.species == ["NO2", "NO", "O3"]


def test_hours_case_without_a_case_is_refused():
    with pytest.raises(ValueError, match="hours='case' needs case"):
        read_results(RESULT_DIR)


def test_a_missing_case_hour_is_named(case, tmp_path):
    result_dir = tmp_path / "RESULT_SOUTHKENSINGTON"
    shutil.copytree(RESULT_DIR, result_dir)
    (result_dir / "RUES_PAR_HEURE" / "Rues_2014010701.dat").unlink()
    with pytest.raises(ValueError, match=r"RUES_PAR_HEURE.*2014-01-07T01:00"):
        read_results(result_dir, case=case)


def test_an_unknown_hours_value_is_named(case):
    with pytest.raises(ValueError, match="hours must be 'case' or 'all'"):
        read_results(RESULT_DIR, case=case, hours="future")


# ------------------------------------------------------------ SIRANE: values and units

def test_negative_zero_reads_as_a_plain_positive_zero(case):
    results = read_results(RESULT_DIR, case=case)
    no2 = results.c_in["NO2"]
    assert no2[0, 0] == 0.0
    assert math.copysign(1.0, no2[0, 0]) == 1.0        # not -0.0


def test_c_in_and_c_above_are_micrograms_to_kg_per_m3(case):
    results = read_results(RESULT_DIR, case=case)
    np.testing.assert_allclose(results.c_in["O3"][:, 0], 1459.33e-9, rtol=1e-9)
    np.testing.assert_allclose(results.c_above["O3"][:, 0], 3.28e-9, rtol=1e-9)


def test_exchange_velocity_pinned_against_sirane_s_own_output(case):
    """SIRANE's own printed `u_d` equals `Sigma_wH * SIRANE_EXCHANGE` (`sigma_w /
    (sqrt(2) pi)`) to the two decimals it prints -- NOT `Sigma_wH / sqrt(2 pi)`, the other
    reading of Soulhac et al. (2011) Eq. 5."""
    results = read_results(RESULT_DIR, case=case)
    right = np.round(results.sigma_w_roof * SIRANE_EXCHANGE, 2)
    np.testing.assert_array_equal(right, results.u_exchange)
    wrong = np.round(results.sigma_w_roof / math.sqrt(2.0 * math.pi), 2)
    assert not np.array_equal(wrong, results.u_exchange)


def test_meteo_from_resul_meteo_dat(case):
    results = read_results(RESULT_DIR, case=case)
    assert set(results.meteo) == {
        "u_star", "sigma_theta", "h_abl", "lmo", "wind_speed", "wind_dir_from_deg",
        "temperature",
    }
    for arr in results.meteo.values():
        assert arr.shape == (2, 46)
    np.testing.assert_allclose(results.meteo["u_star"], 0.14)
    np.testing.assert_allclose(results.meteo["sigma_theta"], math.radians(9.85))
    np.testing.assert_allclose(results.meteo["h_abl"], 114.5)
    np.testing.assert_allclose(results.meteo["lmo"], 100.0)
    np.testing.assert_allclose(results.meteo["wind_speed"], 1.0)
    np.testing.assert_allclose(results.meteo["wind_dir_from_deg"], 315.0)
    np.testing.assert_allclose(results.meteo["temperature"], 277.15)


def test_error_messages_are_named_for_read_results_not_read_case(case, tmp_path):
    """The hourly/meteo table helpers (`_read_table`, `_column`, `_float`) are shared with
    `read_sirane_case`, which hard-codes their errors' prefix to `read_case:`; a `who`
    argument threads `read_results:` through instead when they are reached via this
    reader."""
    result_dir = tmp_path / "RESULT_SOUTHKENSINGTON"
    shutil.copytree(RESULT_DIR, result_dir)
    meteo_path = result_dir / "METEO" / "Resul_Meteo.dat"
    meteo_path.write_text(
        meteo_path.read_text(encoding="latin-1").replace("Ustar", "UstarX"),
        encoding="latin-1",
    )
    with pytest.raises(ValueError,
                       match=r"^read_results: Resul_Meteo\.dat has no 'Ustar' column"):
        read_results(result_dir, case=case)


# -------------------------------------------------------------------------- dispatch

def test_read_results_refuses_a_path_that_is_neither(tmp_path):
    with pytest.raises(ValueError, match="RUES_PAR_HEURE.*case"):
        read_results(tmp_path)


# ----------------------------------------------------------------------- MUNICH results

def test_munich_results_round_trip_against_the_real_excerpt(tmp_path):
    """`munich_paris_excerpt` (see its NOTICE.md) is 4 real streets/5 junctions of MUNICH's
    published Le Perreux-sur-Marne test case, 3 hours, one species (NO2); the excerpt ships
    no `results/` of its own (MUNICH writes those when it runs, not this repo), so this
    fabricates one `NO2.bin` of known values and reads it back through the same `case` the
    real deck files produce."""
    case = read_case(MUNICH_EXCERPT)
    assert case.species == ["NO2"]
    assert case.street_ids == ["1", "3", "8", "11"]
    n_hours, n_streets = len(case.times), len(case.street_ids)
    assert (n_hours, n_streets) == (3, 4)

    # Micrograms/m3, MUNICH's own concentration unit -- distinct per (hour, street).
    values = (np.arange(n_hours * n_streets, dtype=np.float64) + 1.0).reshape(
        n_hours, n_streets
    ) * 12.5
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    values.astype("<f4").tofile(result_dir / "NO2.bin")

    results = read_results(result_dir, case=case)
    assert results.source == "munich"
    assert results.species == ["NO2"]
    assert results.street_ids == case.street_ids       # street order, not just count
    assert results.times == [case.start + timedelta(seconds=t) for t in case.times]
    assert results.c_in["NO2"].shape == (n_hours, n_streets)
    np.testing.assert_allclose(results.c_in["NO2"], values / 1e9, rtol=1e-6)   # ug/m3 -> kg/m3
    assert results.c_above == {}
    assert results.u_canyon is None and results.sigma_w_roof is None
    assert results.u_exchange is None
    assert results.meteo == {}


def test_munich_results_round_trip(tmp_path):
    species = ("NO2", "O3")
    net, case = _munich_case(species=species, n_hours=3)
    n_streets = len(net.streets)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    values = {}
    for i, sp in enumerate(species):
        arr = np.arange(3 * n_streets, dtype=np.float64).reshape(3, n_streets) * (i + 1) * 10.0
        arr.astype("<f4").tofile(result_dir / f"{sp}.bin")
        values[sp] = arr

    results = read_results(result_dir, case=case)
    assert results.source == "munich"
    assert results.times == [case.start + timedelta(seconds=t) for t in case.times]
    assert results.street_ids == case.street_ids
    assert results.species == list(species)
    for sp in species:
        assert results.c_in[sp].shape == (3, n_streets)
        np.testing.assert_allclose(results.c_in[sp], values[sp] / 1e9, rtol=1e-6)
    assert results.c_above == {}
    assert results.u_canyon is None
    assert results.sigma_w_roof is None
    assert results.u_exchange is None
    assert results.meteo == {}


def test_munich_results_refuse_an_hours_selection(tmp_path):
    _, case = _munich_case()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    with pytest.raises(ValueError, match="read_results: hours='all' applies to a SIRANE"):
        read_results(result_dir, case=case, hours="all")


def test_munich_results_needs_a_species_file(tmp_path):
    _, case = _munich_case()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="NO2.bin"):
        read_results(result_dir, case=case)


def test_munich_results_size_mismatch_is_named(tmp_path):
    net, case = _munich_case(n_hours=2)
    n_streets = len(net.streets)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    np.zeros(n_streets, dtype="<f4").tofile(result_dir / "NO2.bin")   # 1 hour's worth, not 2
    with pytest.raises(ValueError, match=r"expected n_hours x n_streets = 2 x \d+"):
        read_results(result_dir, case=case)


def test_munich_results_needs_case_start(tmp_path):
    net, _ = munich_idealised()
    case = StreetCase.synthetic(
        net, species=("NO2",), times=[0.0],
        meteo=dict(wind_dir_from_deg=200.0, wind_speed=5.0),
        emissions=0.0, background=0.0,
    )
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    with pytest.raises(ValueError, match="case.start"):
        read_results(result_dir, case=case)
