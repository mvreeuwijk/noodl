"""SIRANE decks written from the format-neutral `StreetCase` (`write_case(format="sirane")`)
and sweeps of them (`write_sweep`): every written deck is checked by reading it back with
`read_case`, SIRANE's own reader's counterpart in noodl physics."""
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from noodl.apps.street_aq import (
    StreetCase,
    munich_idealised,
    read_case,
    write_case,
    write_sweep,
)
from noodl.apps.street_aq.network import Street, StreetNetwork

DATA = Path(__file__).resolve().parents[2] / "data" / "street" / "sirane_south_kensington"
MASTER = DATA / "Donnees_SouthKensington.dat"
START = datetime(2014, 1, 7)


@pytest.fixture(scope="module")
def sk():
    return read_case(MASTER)


def _synthetic(n_hours=2, *, species=("NO2",), emissions=1e-3, background=2e-8, **meteo):
    net, _ = munich_idealised()
    base = dict(wind_dir_from_deg=45.0, wind_speed=5.0)
    base.update(meteo)
    return StreetCase.synthetic(
        net, species=species, times=[3600.0 * k for k in range(n_hours)], meteo=base,
        emissions=emissions, background=background, start=START,
    )


def _lines(path: Path) -> list[str]:
    return path.read_bytes().decode("latin-1").split("\r\n")


def _assert_same_case(back: StreetCase, case: StreetCase, *, junctions_renamed=False):
    assert back.source == "sirane"
    assert back.times == case.times and back.start == case.start
    assert back.species == case.species
    net, net_back = case.network, back.network
    assert len(net_back.streets) == len(net.streets)
    rename = dict(zip(case.junction_ids, back.junction_ids, strict=True))
    if not junctions_renamed:
        assert back.junction_ids == case.junction_ids
    for a, b in zip(net.streets, net_back.streets, strict=True):
        assert (rename[a.u], rename[a.v]) == (b.u, b.v)
        for attribute in ("length", "width", "height", "z0_b"):
            assert getattr(b, attribute) == pytest.approx(getattr(a, attribute), abs=1e-9)
    for j, j_back in rename.items():
        assert (net_back.x[j_back], net_back.y[j_back]) == (net.x[j], net.y[j])
    for key in ("wind_dir_from_deg", "wind_speed"):
        np.testing.assert_array_equal(back.meteo[key], case.meteo[key])
    if "temperature" in case.meteo:
        np.testing.assert_allclose(back.meteo["temperature"], case.meteo["temperature"],
                                   rtol=0, atol=5e-3)
    np.testing.assert_allclose(back.emissions, case.emissions, rtol=1e-12, atol=0)
    np.testing.assert_allclose(back.background, case.background, rtol=1e-12, atol=0)


# --------------------------------------------------------------------------- round trips

def test_the_south_kensington_deck_writes_back_to_the_same_case(sk, tmp_path):
    out = write_case(tmp_path / "SOUTH_KENSINGTON", sk, format="sirane")
    assert out == tmp_path / "SOUTH_KENSINGTON"
    back = read_case(out / "Donnees.dat")
    _assert_same_case(back, sk)
    # What only a SIRANE case carries survives too: one-sided streets (their own HG/HD are
    # written back), site files, the physics switches and the species' deposition flags.
    assert back.native["one_sided"] == sk.native["one_sided"]
    assert back.native["streets"] == sk.native["streets"]
    assert back.native["site_disp"] == sk.native["site_disp"]
    assert back.native["site_meteo"] == sk.native["site_meteo"]
    assert back.native["physics"] == sk.native["physics"]
    assert back.native["species_table"] == sk.native["species_table"]
    assert back.native["meteo_raw"] == sk.native["meteo_raw"]
    for key in ("U_MIN", "SIGMA_V_MIN", "SIGMA_W_MIN", "H_R", "CHAPMAN", "RATIO_GRILLE",
                "RETRO_BUFF", "Z0D_BAT"):
        assert float(back.native["options"][key]) == float(sk.native["options"][key])


def test_a_synthetic_case_writes_and_re_reads(tmp_path):
    case = _synthetic(3, temperature=281.15)
    write_case(tmp_path / "deck", case, format="sirane")
    back = read_case(tmp_path / "deck" / "Donnees.dat")
    # SIRANE's node ids are whole numbers: the lettered junctions are numbered in order.
    _assert_same_case(back, case, junctions_renamed=True)
    assert back.junction_ids == [str(i + 1) for i in range(len(case.junction_ids))]
    assert back.street_ids == [str(i) for i in range(len(case.street_ids))]
    assert back.native["one_sided"] == []
    assert back.native["site_disp"]["LATITUDE"] == 51.5
    assert back.native["site_disp"]["ZDISPL"] == 13.0
    assert back.native["site_meteo"]["ZDISPL"] == 0.0
    assert back.native["site_meteo"]["ALTITUDE"] == 10.0


def test_emissions_varying_by_hour_get_one_street_file_per_distinct_hour(tmp_path):
    net, _ = munich_idealised()
    emissions = np.zeros((3, len(net.streets), 1))
    emissions[0, 10, 0] = 2e-3
    emissions[1, 3, 0] = 5e-4
    emissions[2, 10, 0] = 2e-3
    case = _synthetic(3, emissions=emissions)
    write_case(tmp_path / "deck", case, format="sirane")
    evolution = _lines(tmp_path / "deck" / "EMISSIONS" / "Emissions_Lin_Surf.dat")
    assert [row.split("\t")[1] for row in evolution[1:4]] == [
        "EMISSIONS/EMIS_LIN/Emis_Rues_0.dat", "EMISSIONS/EMIS_LIN/Emis_Rues_1.dat",
        "EMISSIONS/EMIS_LIN/Emis_Rues_0.dat"]
    back = read_case(tmp_path / "deck" / "Donnees.dat")
    np.testing.assert_allclose(back.emissions, emissions, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------- formats and units

def test_meteo_file_is_the_south_kensington_deck_format(tmp_path):
    write_case(tmp_path / "deck", _synthetic(wind_dir_from_deg=315.0, wind_speed=10.0),
               format="sirane")
    lines = _lines(tmp_path / "deck" / "METEO" / "Meteo_change_1h.dat")
    assert lines[0] == "Date\tU\tDir\tTemp\tPrecip\tCld\tFichier"
    # fprintf('%s\t', date); fprintf('%2.1f\t %i\t %3.2f\t %2.1f\t %i\t %s\r\n', ...)
    assert lines[1] == "07/01/2014 00:00\t10.0\t 315\t 4.00\t 0.0\t 5\t NULL"
    assert lines[2] == "07/01/2014 01:00\t10.0\t 315\t 4.00\t 0.0\t 5\t NULL"


def test_direction_is_written_as_degrees_from_without_a_flip(tmp_path):
    write_case(tmp_path / "deck", _synthetic(wind_dir_from_deg=90.0), format="sirane")
    row = _lines(tmp_path / "deck" / "METEO" / "Meteo_change_1h.dat")[1].split("\t")
    assert row[2].strip() == "90"


def test_masses_are_written_in_sirane_units(tmp_path):
    net, _ = munich_idealised()
    emissions = np.zeros((2, len(net.streets), 1))
    emissions[:, 4, 0] = 1e-3  # kg/s
    write_case(tmp_path / "deck", _synthetic(emissions=emissions, background=4e-8),
               format="sirane")
    rows = _lines(tmp_path / "deck" / "EMISSIONS" / "EMIS_LIN" / "Emis_Rues.dat")
    assert rows[0] == "Id\tNO2"
    assert rows[5] == "4\t1"  # g/s
    assert rows[1] == "0\t0"
    fond = _lines(tmp_path / "deck" / "FOND" / "Concentration_Fond.dat")
    assert fond[0] == "Date\tNO2"
    assert fond[1] == "07/01/2014 00:00\t40"  # micrograms/m3


def test_the_background_covers_whole_days_around_the_period(tmp_path):
    case = StreetCase.synthetic(
        munich_idealised()[0], species=["NO2"], times=[0.0, 3600.0],
        meteo=dict(wind_dir_from_deg=0.0, wind_speed=5.0), emissions=0.0,
        background=np.array([1e-9, 3e-9]), start=datetime(2014, 1, 7, 23),
    )
    write_case(tmp_path / "deck", case, format="sirane")
    fond = [row for row in _lines(tmp_path / "deck" / "FOND" / "Concentration_Fond.dat")
            if row]
    assert len(fond) == 1 + 48
    assert fond[1] == "07/01/2014 00:00\t1"
    assert fond[24] == "07/01/2014 23:00\t1"
    assert fond[25] == "08/01/2014 00:00\t3"
    assert fond[48] == "08/01/2014 23:00\t3"
    back = read_case(tmp_path / "deck" / "Donnees.dat")
    np.testing.assert_allclose(back.background[:, 0, 0], [1e-9, 3e-9], rtol=1e-12)


def test_a_passive_tracer_is_an_existing_species_with_chemistry_and_deposition_off(tmp_path):
    write_case(tmp_path / "deck", _synthetic(), format="sirane")
    master = _lines(tmp_path / "deck" / "Donnees.dat")
    assert "Activation du modele chimique de Chapman [0/1] = 0" in master
    species = [row.split("\t") for row in _lines(tmp_path / "deck" / "ESPECES" / "Especes.dat")
               if row]
    assert species[0][:4] == ["Id", "Flag", "Mmolaire", "Vdepot"]
    assert species[1][:5] == ["NO2", "1", "46", "0", "0"]
    assert all(row[1] == "0" for row in species[2:])  # every other species inactive
    back = read_case(tmp_path / "deck" / "Donnees.dat")
    assert back.species == ["NO2"]
    assert back.native["physics"]["chemistry_on"] is False


def test_options_set_chemistry_plume_and_deposition(tmp_path):
    case = _synthetic(species=("NO2", "NO", "O3"))
    write_case(tmp_path / "deck", case, format="sirane",
               options={"chapman": 1, "plume": 0, "deposition": 1, "latitude": 48.85,
                        "U_MIN": 0.5})
    back = read_case(tmp_path / "deck" / "Donnees.dat")
    assert back.native["physics"]["chemistry_on"] is True
    assert back.native["physics"]["plume_on"] is False
    assert all(row["Vdepot"] == "1" for row in back.native["species_table"].values())
    assert back.native["site_disp"]["LATITUDE"] == 48.85
    assert back.native["options"]["U_MIN"] == "0.5"


def test_every_master_label_is_one_sirane_defines_in_french(tmp_path):
    from noodl.apps.street_aq._sirane_files import _MASTER_LABELS, _SITE_LABELS

    write_case(tmp_path / "deck", _synthetic(), format="sirane")
    french = {labels[0] for labels in _MASTER_LABELS.values()}
    for line in _lines(tmp_path / "deck" / "Donnees.dat"):
        if line and not line.startswith("/"):
            assert line.split(" = ")[0] in french, line
    french_site = {labels[0] for labels in _SITE_LABELS.values()}
    for name in ("RESEAU/Site_Disp.dat", "METEO/Site_Meteo.dat"):
        for line in _lines(tmp_path / "deck" / name):
            if line and not line.startswith("/"):
                assert line.split(" = ")[0] in french_site, line


def test_folders_default_to_paths_relative_to_the_deck_s_parent(tmp_path):
    write_case(tmp_path / "my_deck", _synthetic(), format="sirane")
    master = _lines(tmp_path / "my_deck" / "Donnees.dat")
    assert "Repertoire des donnees d'entree = my_deck" in master
    assert "Repertoire d'ecriture des resultats = my_deck/RESULT" in master
    write_case(tmp_path / "a" / "b", _synthetic(), format="sirane",
               options={"input_dir": "a/b", "result_dir": "out"})
    master = _lines(tmp_path / "a" / "b" / "Donnees.dat")
    assert "Repertoire des donnees d'entree = a/b" in master
    assert "Repertoire d'ecriture des resultats = out" in master
    assert (tmp_path / "out" / "RUES_PAR_HEURE").is_dir()  # below SIRANE's working dir


def test_an_input_dir_that_is_not_where_the_deck_is_written_is_refused(tmp_path):
    with pytest.raises(ValueError, match="trailing part of out_dir"):
        write_case(tmp_path / "other", _synthetic(), format="sirane",
                   options={"input_dir": "a/b"})


def test_the_result_folders_are_created_in_advance(tmp_path):
    write_case(tmp_path / "deck", _synthetic(), format="sirane")
    # The folders SIRANE's own output log lists under the results folder, all "existe deja".
    for name in ("METEO", "RECEPT", "RECEPT_STAT", "RUES_PAR_HEURE", "RUES_PAR_RUE",
                 "RUES_STAT", "GRILLE", "GRILLE_STAT", "IMAGES", "IMAGES_STAT"):
        assert (tmp_path / "deck" / "RESULT" / name).is_dir(), name


def _labels(path: Path) -> list[str]:
    return [line.split("=")[0].strip()
            for line in path.read_bytes().decode("latin-1").splitlines()
            if line.strip() and not line.startswith("/") and "=" in line]


PLUME_LABEL = "Prise en compte des rues-panaches [0/1]"
"""`B_PANACHE`'s French description, verbatim from SIRANE v2.1's `Don_Defaut_FR.dat` (the
one label a sweep writes that the South Kensington deck does not set)."""


def test_written_labels_are_the_south_kensington_deck_s_own(tmp_path):
    reference = set(_labels(MASTER))
    reference_site = set(_labels(DATA / "RESEAU" / "Site_Disp.dat"))
    assert reference_site == set(_labels(DATA / "METEO" / "Site_Meteo.dat"))
    write_case(tmp_path / "deck", _synthetic(), format="sirane")
    written = _labels(tmp_path / "deck" / "Donnees.dat")
    assert set(written) <= reference, set(written) - reference
    for name in ("RESEAU/Site_Disp.dat", "METEO/Site_Meteo.dat"):
        assert set(_labels(tmp_path / "deck" / name)) == reference_site
    # Switching the street-plume model off needs the one label the deck lacks.
    write_case(tmp_path / "off", _synthetic(), format="sirane", options={"plume": 0})
    extra = set(_labels(tmp_path / "off" / "Donnees.dat")) - reference
    assert extra == {PLUME_LABEL}


def test_carried_over_settings_outside_sirane_s_range_are_refused(sk, tmp_path):
    from dataclasses import replace

    native = dict(sk.native, options=dict(sk.native["options"], U_MIN="9"))
    with pytest.raises(ValueError, match="U_MIN.*range"):
        write_case(tmp_path / "deck", replace(sk, native=native), format="sirane")
    native = dict(sk.native, site_disp=dict(sk.native["site_disp"], ZDISPL=60.0))
    with pytest.raises(ValueError, match="ZDISPL.*range"):
        write_case(tmp_path / "deck", replace(sk, native=native), format="sirane")


# ------------------------------------------------------------------------------ refusals

@pytest.mark.parametrize("kwargs, options, match", [
    (dict(species=("tracer",)), None, "not SIRANE species"),
    (dict(wind_dir_from_deg=22.5), None, "whole number of degrees"),
    (dict(wind_speed=5.05), None, "multiple of 0.1"),
    ({}, {"chapman": 1}, "lack"),
    ({}, {"chemistry": 1}, "not understood"),
    ({}, {"plume": 2}, "plume.*range"),
    ({}, {"latitude": 95.0}, "LATITUDE.*range"),
    ({}, {"measurement_height": 120.0}, "ALTITUDE.*range"),
    ({}, {"U_MIN": 7}, "U_MIN.*range"),
    ({}, {"RATIO_GRILLE": 1.5}, "whole number"),
])
def test_what_sirane_cannot_express_is_named(tmp_path, kwargs, options, match):
    with pytest.raises(ValueError, match=match):
        write_case(tmp_path / "deck", _synthetic(**kwargs), format="sirane", options=options)


def test_meteo_varying_between_streets_is_refused(tmp_path):
    net, _ = munich_idealised()
    direction = np.full((2, len(net.streets)), 45.0)
    direction[:, 3] = 50.0
    with pytest.raises(ValueError, match="varies between streets"):
        write_case(tmp_path / "deck", _synthetic(wind_dir_from_deg=direction),
                   format="sirane")


def test_a_street_length_that_is_not_its_geometry_is_refused(tmp_path):
    net = StreetNetwork(streets=[Street("a", "p", "q", 150.0, 20.0, 20.0)],
                        x={"p": 0.0, "q": 100.0}, y={"p": 0.0, "q": 0.0})
    case = StreetCase.synthetic(net, species=["NO2"], times=[0.0],
                                meteo=dict(wind_dir_from_deg=0.0, wind_speed=5.0),
                                emissions=0.0, background=0.0, start=START)
    with pytest.raises(ValueError, match="length"):
        write_case(tmp_path / "deck", case, format="sirane")


def test_sirane_needs_a_start_date(tmp_path):
    net, _ = munich_idealised()
    case = StreetCase.synthetic(net, species=["NO2"], times=[0.0],
                                meteo=dict(wind_dir_from_deg=0.0, wind_speed=5.0),
                                emissions=0.0, background=0.0)
    with pytest.raises(ValueError, match="start"):
        write_case(tmp_path / "deck", case, format="sirane")


# --------------------------------------------------------------------------------- sweeps

def test_a_sweep_writes_one_deck_per_run_and_a_manifest_row_for_each(tmp_path):
    case = _synthetic()
    directions, speeds, sources = [0, 90, 225], [2, 5], ["1", "11"]
    out = write_sweep(tmp_path / "sweep", case, directions_deg=directions, speeds=speeds,
                      sources=sources)
    decks = sorted(p.name for p in (out / "decks").iterdir())
    n_runs = len(directions) * len(speeds) * len(sources) * 2
    assert len(decks) == n_runs
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in decks)
    assert "d225_u5p0_s10_plume_off" in decks  # street "11" is SIRANE's street 10
    assert all((out / "decks" / name / "Donnees.dat").is_file() for name in decks)
    assert sorted(p.name for p in out.iterdir()) == ["decks", "runs.csv"]
    csv = [row for row in _lines(out / "runs.csv") if row]
    assert len(csv) == 1 + n_runs
    assert csv[0].startswith("run_id,direction_deg,speed_m_s,source_index,source_street")
    assert sorted(row.split(",")[0] for row in csv[1:]) == decks
    row = next(r for r in csv if r.startswith("d225_u5p0_s10_plume_off,"))
    assert row.split(",")[1:6] == ["225", "5", "10", "11", "plume_off"]


def test_a_base_case_sirane_cannot_take_leaves_no_files(tmp_path):
    net = StreetNetwork(streets=[Street("a", "p", "q", 150.0, 20.0, 20.0)],
                        x={"p": 0.0, "q": 100.0}, y={"p": 0.0, "q": 0.0})
    case = StreetCase.synthetic(net, species=["NO2"], times=[0.0],
                                meteo=dict(wind_dir_from_deg=0.0, wind_speed=5.0),
                                emissions=0.0, background=0.0, start=START)
    with pytest.raises(ValueError, match="length"):
        write_sweep(tmp_path / "sweep", case, directions_deg=[0], speeds=[5])
    assert not (tmp_path / "sweep").exists()


def test_every_sweep_deck_re_reads_as_its_run(tmp_path):
    case = _synthetic(temperature=278.15)
    out = write_sweep(tmp_path / "sweep", case, directions_deg=[45, 315], speeds=[2.5],
                      sources=["4"], variants={"base": {}})
    for direction in (45, 315):
        deck = out / "decks" / f"d{direction:03d}_u2p5_s03_base" / "Donnees.dat"
        back = read_case(deck)
        assert back.times == [0.0, 3600.0] and back.start == START
        assert back.species == ["NO2"]
        assert np.all(back.meteo["wind_dir_from_deg"] == direction)
        assert np.all(back.meteo["wind_speed"] == 2.5)
        np.testing.assert_allclose(back.meteo["temperature"], 278.15, atol=5e-3)
        expected = np.zeros_like(back.emissions)
        expected[:, 3, 0] = 1e-3
        np.testing.assert_allclose(back.emissions, expected, rtol=1e-12, atol=0)
        assert np.all(back.background == 0.0)
        assert back.native["physics"]["chemistry_on"] is False
        assert all(row["Vdepot"] == "0" for row in back.native["species_table"].values())
        master = _lines(deck)
        run = deck.parent.name
        assert f"Repertoire des donnees d'entree = decks/{run}" in master
        assert f"Repertoire d'ecriture des resultats = decks/{run}/RESULT" in master


def test_a_sweep_over_every_street_of_a_sirane_case_keeps_its_own_settings(sk, tmp_path):
    out = write_sweep(tmp_path / "sweep", sk, directions_deg=[135], speeds=[1],
                      variants={"plume_off": {"plume": 0}})
    assert len(list((out / "decks").iterdir())) == len(sk.street_ids)
    back = read_case(out / "decks" / "d135_u1p0_s04_plume_off" / "Donnees.dat")
    assert back.native["one_sided"] == sk.native["one_sided"]
    assert back.native["site_disp"] == sk.native["site_disp"]
    assert back.native["physics"]["plume_on"] is False
    assert back.native["physics"]["chemistry_on"] is False  # the sweep's chapman=0
    assert back.species == ["NO2"]
    assert back.emissions[1, 4, 0] == pytest.approx(1e-3)


def test_a_chemistry_sweep_carries_all_three_chapman_species(tmp_path):
    out = write_sweep(tmp_path / "sweep", _synthetic(), directions_deg=[0], speeds=[5],
                      sources=["1"], species="O3", chapman=1, variants={"on": {}})
    back = read_case(out / "decks" / "d000_u5p0_s00_on" / "Donnees.dat")
    assert back.species == ["NO2", "NO", "O3"]
    assert back.native["physics"]["chemistry_on"] is True
    assert back.emissions[0, 0].tolist() == [0.0, 0.0, pytest.approx(1e-3)]


@pytest.mark.parametrize("kwargs, match", [
    (dict(variants={"plume on": {}}), "variant name"),
    (dict(directions_deg=[0, 360]), "repeat"),
    (dict(directions_deg=[22.5]), "whole degrees"),
    (dict(speeds=[0.0]), "positive"),
    (dict(sources=["nope"]), "not in base_case"),
    (dict(sources="all"), "unit_impulse"),
    (dict(chapman=1, species="CO"), "NO2, NO or O3"),
    (dict(variants={"ok": {}, "bad": {"plume": 3}}), "plume.*range"),
    (dict(variants={"bad": {"bogus": 1}}), "not understood"),
    (dict(variants={"bad": {"input_dir": "elsewhere"}}), "may not set"),
    (dict(variants={"bad": {"latitude": -91}}), "LATITUDE"),
    (dict(variants={"bad": {"chapman": 1}}), "lack"),
])
def test_sweep_arguments_are_checked_before_anything_is_written(tmp_path, kwargs, match):
    arguments = dict(directions_deg=[0], speeds=[5.0])
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        write_sweep(tmp_path / "sweep", _synthetic(), **arguments)
    assert not (tmp_path / "sweep").exists()


def test_run_ids_distinguish_every_speed_to_a_tenth(tmp_path):
    out = write_sweep(tmp_path / "sweep", _synthetic(), directions_deg=[0],
                      speeds=[1.0, 1.1, 10.0], sources=["1"], variants={"v": {}})
    assert sorted(p.name for p in (out / "decks").iterdir()) == [
        "d000_u10p0_s00_v", "d000_u1p0_s00_v", "d000_u1p1_s00_v"]
    assert math.isclose(read_case(out / "decks" / "d000_u1p1_s00_v" / "Donnees.dat")
                        .meteo["wind_speed"][0, 0], 1.1)
