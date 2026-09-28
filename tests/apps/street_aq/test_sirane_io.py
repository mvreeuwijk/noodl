"""SIRANE v2.1 decks read into the format-neutral `StreetCase`: the South Kensington deck
(`tests/data/street/sirane_south_kensington`, see its NOTICE) in its own French labels and
rewritten with SIRANE's English ones, the date formats SIRANE's files mix, and what the
reader refuses by name."""
import math
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.street_aq import build_model
from noodl.apps.street_aq._sirane_files import parse_date
from noodl.apps.street_aq.case import StreetCase, drivers_at, read_case
from noodl.apps.street_aq.closures import resolve

DATA = Path(__file__).resolve().parents[2] / "data" / "street" / "sirane_south_kensington"
MASTER = DATA / "Donnees_SouthKensington.dat"

# The South Kensington master with every label in SIRANE's English wording (the
# descriptions of its `Don_Defaut_EN.dat`), same values, same order.
EN_MASTER = """\
/******************************************
/** Data for SIRANE **
/******************************************
Input data folder = SOUTH_KENSINGTON
Begin date = 07/01/2014 00:00:00
End date = 07/01/2014 01:00:00
Pollutants file = ESPECES/Especes.dat
Sources groups file = ESPECES/SrceGroup.dat
Activation of the Chapman chemical model [0/1] = 1
NO emissions in equivalent NO2 [0/1] = 0
Point source file = EMISSIONS/Sources_Point.dat
Emissions file = EMISSIONS/Emissions_Lin_Surf.dat
Number of lineic modulations = 1
Background concentration file = FOND/Concentration_Fond.dat
Streets subdivision on the retrotrajectory grid [0/1] = 0
Network file type [0/1/2] = 2
Network of streets file = RESEAU/SIRANE_FINAL
Dispersion site data file = RESEAU/Site_Disp.dat
Building surface roughness [m] = 1
Canopy height [m] = 20.0
Meteorological conditions [0/1/2/3] = 0
Meteorological file = METEO/Meteo_change_1h.dat
Meteorological site data file = METEO/Site_Meteo.dat
Street mean velocity and sigma_w provided [0/1] = 0
Minimum wind velocity [m/s] = 0
Minimum sigma_v [m/s] = 0
Minimum sigma_w [m/s] = 0
Meteorological grid definition file = GRILLES/Grille_Meteo_L93.dat
Output grid definition file = GRILLES/Grille_Sortie_L93.dat
Surface emissions grid definition file = GRILLES/Grille_Emis_Surf_L93.dat
Verbose level [0/1/2] = 2
Statistics calculation [0/1] = 1
Point receptor file = RECEPTEURS/recep.dat
Daily averaged concentrations calculation [0/1] = 0
Grid results output [0/1/2] = 2
Writing deposition [0/1] = 0
Streets results output [0/1] = 1
Streets results format [0/1] = 0
Street intersection concentration calculation [0/1] = 0
Grid results format [0/1/2/3/4] = 4
Image results output [0/1/2] = 1
Pollutants colormap file = COLORMAPS/Colormaps_especes.dat
Exceedence colormap file = COLORMAPS/Colormap_NB.dat
Results folder = RESULT_SOUTHKENSINGTON
Percentiles file = STATISTIQUES/Percentiles.dat
Thresholds file = STATISTIQUES/Seuils.dat
Ratio meteo and retrotrajectory grid = 1
Buffer zone in cells for the retrotrajectory calculation = 0
Streets subdivision on the retrotrajectory grid [0/1] = 0
"""

EN_SITE_METEO = """\
/ Meteo site :
Latitude [deg] = 51.494150
Height over ground [m] = 10.0
Aerodynamic roughness [m] = 1.0
Displacement height [m] = 0.0
Albedo = 1
Emissivity = 0.88
Priestley-Taylor coefficient = 0.5
"""


@pytest.fixture(scope="module")
def case() -> StreetCase:
    return read_case(MASTER)


def _deck_copy(tmp_path: Path) -> Path:
    """A writable copy of the deck (inputs only) -- the fixture itself is never modified."""
    root = tmp_path / "deck"
    shutil.copytree(DATA, root, ignore=shutil.ignore_patterns("RESULT_*", "NOTICE.md"))
    return root


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="latin-1")
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new), encoding="latin-1")


# ------------------------------------------------------------------------------ dates

@pytest.mark.parametrize("text, expected", [
    ("07/01/2014 00:00:00", datetime(2014, 1, 7, 0)),      # master file
    ("07/01/2014 01:00", datetime(2014, 1, 7, 1)),         # meteo and background files
    ("7/1/2014 00:00", datetime(2014, 1, 7, 0)),           # Emissions_Lin_Surf.dat
    ("1/1/2014 0:00", datetime(2014, 1, 1, 0)),            # point-source series
    ("2014/01/07 01:00:00", datetime(2014, 1, 7, 1)),      # RUES_PAR_RUE results
    ("31/12/2004 23:00:00", datetime(2004, 12, 31, 23)),   # day first, not month first
])
def test_each_sirane_date_format(text, expected):
    assert parse_date(text, where="test") == expected


@pytest.mark.parametrize("text", ["2014-01-07 00:00", "07/01/14 00:00", "07/01/2014"])
def test_an_unknown_date_format_is_named(text):
    with pytest.raises(ValueError, match="test.*not a SIRANE date"):
        parse_date(text, where="test")


# ---------------------------------------------------------------------------- reading

def test_french_master_reads_the_network(case):
    assert case.source == "sirane"
    assert len(case.street_ids) == 46 and len(case.network.junctions) == 36
    assert case.street_ids == [str(i) for i in range(46)]
    assert sorted(case.junction_ids, key=int) == [str(i) for i in range(1, 37)]
    assert case.junction_ids == case.network.junctions
    street = case.network.streets[0]          # NDDEB 1 -> NDFIN 7, WG = WD = 9
    assert (street.u, street.v) == ("1", "7")
    assert street.width == pytest.approx(18.0)
    assert street.height == pytest.approx(0.5 * (15.762893 + 16.951633))
    assert case.network.x["1"] == pytest.approx(-15911.75, abs=0.01)
    assert case.network.y["7"] == pytest.approx(4991229.32, abs=0.01)
    assert street.length == pytest.approx(
        math.hypot(-15593.12 - -15911.75, 4991229.32 - 4991207.95), abs=0.02
    )
    # SIRANE's building surface roughness (`Z0D_BAT`, 1 m in this deck) on every street.
    assert {s.z0_b for s in case.network.streets} == {1.0}


def test_period_species_and_units(case):
    assert case.start == datetime(2014, 1, 7, 0)
    assert case.times == [0.0, 3600.0]
    assert case.species == ["NO2", "NO", "O3"]              # Flag = 1, in file order
    assert case.emissions.shape == (2, 46, 3) and case.background.shape == (2, 46, 3)
    # Street 4 emits 1 g/s of O3 (x modulation 1): 1e-3 kg/s; nothing else emits.
    expected = np.zeros((2, 46, 3))
    expected[:, 4, 2] = 1e-3
    np.testing.assert_array_equal(case.emissions, expected)
    np.testing.assert_array_equal(case.background, np.zeros((2, 46, 3)))
    assert case.emissions.dtype == np.float64


def test_meteo_is_broadcast_per_street(case):
    assert set(case.meteo) == {"wind_dir_from_deg", "wind_speed", "temperature"}
    for value in case.meteo.values():
        assert value.shape == (2, 46) and value.dtype == np.float64
    np.testing.assert_array_equal(case.meteo["wind_speed"], 1.0)
    np.testing.assert_array_equal(case.meteo["wind_dir_from_deg"], 135.0)
    np.testing.assert_allclose(case.meteo["temperature"], 277.15, rtol=0, atol=1e-12)
    assert case.meteo_junction == {}
    assert case.native["meteo_raw"] == {"precip": [0.0, 0.0], "cloud": [5.0, 5.0]}


def test_native_carries_options_sites_and_one_sided_streets(case):
    options = case.native["options"]
    assert options["CHAPMAN"] == "1" and options["TYPE_FICH_RESEAU"] == "2"
    assert options["FICH_RESEAU"] == "RESEAU/SIRANE_FINAL"
    assert options["DATE_DEB"] == "07/01/2014 00:00:00"
    assert case.native["site_meteo"] == {
        "LATITUDE": 51.49415, "ALTITUDE": 10.0, "Z0D": 1.0, "ZDISPL": 0.0, "ALBEDO": 1.0,
        "EMISSIVITE": 0.88, "PRIESTLEY_TAYLOR": 0.5,
    }
    assert case.native["site_disp"]["ZDISPL"] == 13.0
    assert case.native["one_sided"] == ["3", "14", "42"]      # HG = 0 on these three
    three = case.native["streets"][3]
    assert three["HG"] == 0.0 and three["HD"] == pytest.approx(21.375511)
    assert case.network.streets[3].height == pytest.approx(21.375511 / 2)
    assert case.native["physics"] == {
        "background_on": True, "plume_on": True, "retro_on": True, "dispersion_model": 2,
        "chemistry_on": True,
    }


def test_the_meteo_grid_gives_the_cell_size(case, tmp_path):
    """`native["meteo_grid"]` from the deck's `FICH_GRD_MET` (2 x 2 cells over 1500 m x
    1200 m: 750 m x 600 m cells); `None` when the file is absent; a malformed one is named."""
    assert case.native["meteo_grid"] == {
        "nx": 2, "ny": 2, "xmin": -16000.0, "xmax": -14500.0, "ymin": 4990800.0,
        "ymax": 4992000.0, "dx": 750.0, "dy": 600.0}
    root = _deck_copy(tmp_path)
    grid = root / "GRILLES" / "Grille_Meteo_L93.dat"
    grid.write_text("Nx\tNy\txmin\txmax\tymin\tymax\n2\t2\t0\t0\t0\t10\n")
    with pytest.raises(ValueError, match="empty grid"):
        read_case(root / MASTER.name)
    grid.unlink()
    assert read_case(root / MASTER.name).native["meteo_grid"] is None


def test_english_master_reads_identically(case, tmp_path):
    root = _deck_copy(tmp_path)
    (root / "Donnees_EN.dat").write_text(EN_MASTER, encoding="latin-1")
    (root / "METEO" / "Site_Meteo.dat").write_text(EN_SITE_METEO, encoding="latin-1")
    en = read_case(root / "Donnees_EN.dat")
    assert en.native == case.native
    assert en.street_ids == case.street_ids and en.junction_ids == case.junction_ids
    assert en.network == case.network
    assert (en.times, en.start, en.species) == (case.times, case.start, case.species)
    np.testing.assert_array_equal(en.emissions, case.emissions)
    np.testing.assert_array_equal(en.background, case.background)
    for key in case.meteo:
        np.testing.assert_array_equal(en.meteo[key], case.meteo[key])


def test_hourly_modulation_scales_the_street_emission(tmp_path):
    root = _deck_copy(tmp_path)
    lin = root / "EMISSIONS" / "Emissions_Lin_Surf.dat"
    lines = lin.read_text(encoding="latin-1").splitlines()
    # Hour 01: O3 modulation 2.5 (the third Mod_Lin column is Mod_Lin_0_O3).
    fields = lines[2].split("\t")
    fields[4] = "2.5"
    lines[2] = "\t".join(fields)
    lin.write_text("\n".join(lines) + "\n", encoding="latin-1")
    case = read_case(root / MASTER.name)
    assert case.emissions[0, 4, 2] == pytest.approx(1e-3)
    assert case.emissions[1, 4, 2] == pytest.approx(2.5e-3)


def test_background_is_micrograms_to_kg_per_m3(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / "FOND" / "Concentration_Fond.dat",
             "07/01/2014 01:00\t0\t0\t0", "07/01/2014 01:00\t40\t0\t60")
    case = read_case(root / MASTER.name)
    np.testing.assert_array_equal(case.background[0], 0.0)
    np.testing.assert_allclose(case.background[1, :, 0], 40e-9, rtol=1e-15)
    np.testing.assert_allclose(case.background[1, :, 2], 60e-9, rtol=1e-15)


# --------------------------------------------------------------------- model options

def test_model_options_are_sirane_s_closures(case):
    options = case.model_options()
    # SIRANE's closure set is the sirane preset; the deck adds its own turbulence floors
    # (this deck zeroes both).
    assert options == {"preset": "sirane", "sigma_w_min": 0.0, "sigma_v_min": 0.0}
    resolved = resolve("sirane", {}, "test")
    assert resolved["canyon_wind"] == "bessel_profile"
    assert resolved["roof_exchange"] == "turbulent_velocity"
    assert resolved["junction_routing"] == "non_crossing_streamlines"
    # The exact Gaussian average over the direction spread (Soulhac et al. 2011, Eq. 7),
    # not a fixed quadrature: no node count to choose.
    assert resolved["direction_averaging"] == "exact_gaussian"
    assert resolved["direction_spread"] == "driver"
    assert "n_theta" not in options
    assert resolved["stability"] == "monin_obukhov"
    # No z_ref: noodl physics has no SIRANE meteorological preprocessor, so a SIRANE case
    # is driven with SIRANE's own u* (the `u_star` driver), never noodl's log law.
    assert "z_ref" not in options


def test_model_options_fall_back_to_sirane_s_default_floors(case):
    deck = {k: v for k, v in case.native["options"].items()
            if k not in ("SIGMA_W_MIN", "SIGMA_V_MIN")}
    bare = replace(case, native=dict(case.native, options=deck))
    assert bare.model_options() == {"preset": "sirane"}
    set_ = replace(case, native=dict(case.native, options=dict(deck, SIGMA_W_MIN="0.3",
                                                               SIGMA_V_MIN="0.5")))
    assert set_.model_options() == {"preset": "sirane", "sigma_w_min": 0.3,
                                    "sigma_v_min": 0.5}


def test_the_case_drives_a_model_built_from_its_own_options(case):
    model, _state, _ = build_model(case.network, species=case.species,
                                   meteo="per_street", **case.model_options())
    drivers = drivers_at(case, model, 1)
    # The boundary layer and direction spread SIRANE's own preprocessor derived for this
    # hour (its `Resul_Meteo.dat`), which the deck itself does not carry.
    drivers["h_abl"] = torch.full((46,), 114.5, dtype=torch.float64)
    drivers["u_star"] = torch.full((46,), 0.14, dtype=torch.float64)
    drivers["lmo"] = torch.full((46,), 100.0, dtype=torch.float64)
    drivers["sigma_theta"] = torch.full((36,), math.radians(9.85), dtype=torch.float64)
    out = model.closures[0](None, drivers)
    assert bool(torch.isfinite(out["street.q"]).all())


# ------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("old, new, match", [
    ("Type de fichier de reseau [0/1/2] = 2", "Type de fichier de reseau [0/1/2] = 0",
     "TYPE_FICH_RESEAU"),
    ("Conditions meteorologiques [0/1/2/3] = 0", "Conditions meteorologiques [0/1/2/3] = 1",
     "TYPE_METEO"),
    ("Vitesse moyenne et sigma_w des rues fournis [0/1] = 0",
     "Vitesse moyenne et sigma_w des rues fournis [0/1] = 1", "B_STREET_U_SIGMA_W"),
    ("Emissions de NO en equivalent NO2 [0/1] = 0",
     "Emissions de NO en equivalent NO2 [0/1] = 1", "B_EMIS_NOEQNO2"),
    ("Date de fin = 07/01/2014 01:00:00", "Date de fin = 07/01/2014 03:00:00",
     "Meteo_change_1h.dat.*2014-01-07T02:00"),
    ("Date de fin = 07/01/2014 01:00:00", "Date de fin = 06/01/2014 23:00:00",
     "DATE_FIN.*before"),
])
def test_master_settings_outside_what_the_reader_handles_are_named(tmp_path, old, new,
                                                                    match):
    root = _deck_copy(tmp_path)
    _replace(root / MASTER.name, old, new)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        read_case(root / MASTER.name)


def test_a_missing_mandatory_key_is_named(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / MASTER.name, "Fichier meteo = METEO/Meteo_change_1h.dat\n", "")
    with pytest.raises(ValueError, match="FICH_METEO"):
        read_case(root / MASTER.name)


def test_non_empty_surface_emissions_are_refused(tmp_path):
    root = _deck_copy(tmp_path)
    path = root / "EMISSIONS" / "EMIS_SURF" / "EmisSurf_nulle.dat"
    path.write_text("X\tY\tNO2\tNO\tO3\n-15000\t4991000\t1\t0\t0\n", encoding="latin-1")
    with pytest.raises(NotImplementedError, match="surface emissions"):
        read_case(root / MASTER.name)


def test_a_point_source_emitting_in_the_period_is_refused_wherever_it_sits(tmp_path):
    root = _deck_copy(tmp_path)
    # The deck's one source lies kilometres outside the network; it still counts.
    _replace(root / "EMISSIONS" / "EMIS_PONCT" / "Emis_ponct_nulle.dat",
             "7/1/2014 1:00\t0\t0\t0", "7/1/2014 1:00\t0\t2.5\t0")
    with pytest.raises(NotImplementedError,
                       match="point source 'id' emits at 2014-01-07T01:00"):
        read_case(root / MASTER.name)


def test_a_silent_point_source_is_ignored_even_inside_the_network(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / "EMISSIONS" / "Sources_Point.dat", "-12401.58\t3971179.5",
             "-15500.0\t4991400.0")
    # Emitting outside the period does not matter either.
    _replace(root / "EMISSIONS" / "EMIS_PONCT" / "Emis_ponct_nulle.dat",
             "7/1/2014 2:00\t0\t0\t0", "7/1/2014 2:00\t9\t9\t9")
    assert read_case(root / MASTER.name).emissions[:, 4, 2].tolist() == [1e-3, 1e-3]


def test_a_point_source_series_not_covering_the_period_is_named(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / "EMISSIONS" / "EMIS_PONCT" / "Emis_ponct_nulle.dat",
             "7/1/2014 1:00\t0\t0\t0\n", "")
    with pytest.raises(ValueError, match="Emis_ponct_nulle.dat has no row for"):
        read_case(root / MASTER.name)


def test_a_meteo_override_file_is_refused(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / "METEO" / "Meteo_change_1h.dat", "5\t NULL\n", "5\t hour0.dat\n")
    with pytest.raises(NotImplementedError, match="Fichier"):
        read_case(root / MASTER.name)


def test_a_street_missing_from_the_emission_file_is_named(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / "EMISSIONS" / "EMIS_LIN" / "Emis_Rues.dat", "45\t 0\t0\t0\t", "")
    with pytest.raises(ValueError, match="Emis_Rues.dat.*45"):
        read_case(root / MASTER.name)


def test_duplicate_keys_with_different_values_are_named(tmp_path):
    root = _deck_copy(tmp_path)
    path = root / MASTER.name
    path.write_text(path.read_text(encoding="latin-1") + "Rugosite aerodynamique des "
                    "batiments [m] = 0.5\n", encoding="latin-1")
    with pytest.raises(ValueError, match="Z0D_BAT"):
        read_case(path)


def test_a_key_the_deck_leaves_at_its_default_is_recognised(tmp_path):
    root = _deck_copy(tmp_path)
    path = root / MASTER.name
    path.write_text(path.read_text(encoding="latin-1") + "Seuil sur sigma pour negliger "
                    "une bouffee = 4.0\nSigma threshold to neglect a puff = 4.0\n",
                    encoding="latin-1")
    assert read_case(path).native["options"]["SEUIL_GAUSS"] == "4.0"


def test_a_misspelt_label_is_refused_by_name(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / MASTER.name, "Fichier des especes =", "Fichier des espece =")
    with pytest.raises(ValueError, match="label 'Fichier des espece' is not one SIRANE"):
        read_case(root / MASTER.name)


def test_background_switched_off_zeroes_the_case_background(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / "FOND" / "Concentration_Fond.dat",
             "07/01/2014 01:00\t0\t0\t0", "07/01/2014 01:00\t40\t0\t60")
    path = root / MASTER.name
    path.write_text(path.read_text(encoding="latin-1") + "Prise en compte de la pollution "
                    "de fond [0/1] = 0\nPrise en compte des rues-panaches [0/1] = 0\n"
                    "Modele de diffusion [0/1/2] = 1\n", encoding="latin-1")
    case = read_case(path)
    np.testing.assert_array_equal(case.background, 0.0)
    assert case.native["physics"] == {
        "background_on": False, "plume_on": False, "retro_on": True, "dispersion_model": 1,
        "chemistry_on": True,
    }


@pytest.mark.parametrize("relative, old, new", [
    ("METEO/Meteo_change_1h.dat", "07/01/2014 01:00\t1.0\t 135\t 4.00\t 0.0\t 5\t NULL ",
     "07/01/2014 01:00\t1.0\t 135"),
    ("FOND/Concentration_Fond.dat", "07/01/2014 01:00\t0\t0\t0", "07/01/2014 01:00\t0"),
    ("EMISSIONS/EMIS_LIN/Emis_Rues.dat", "45\t 0\t0\t0\t", "45\t 0"),
])
def test_a_short_table_row_is_named(tmp_path, relative, old, new):
    root = _deck_copy(tmp_path)
    _replace(root / relative, old, new)
    name = relative.rsplit("/", 1)[1]
    with pytest.raises(ValueError, match=rf"{name} line \d+ has \d field\(s\)"):
        read_case(root / MASTER.name)


# ------------------------------------------------------------- the archived results

def _table(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="latin-1").splitlines()
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:] if ln.strip()]


def test_the_archived_run_is_not_the_deck_as_shipped(case):
    """Pins what NOTICE.md records: the archived results were produced with other meteo
    and another sigma_w floor than the deck's own files -- so a comparison must drive noodl
    physics with the results' own meteorology, not the deck's."""
    header, rows = _table(DATA / "RESULT_SOUTHKENSINGTON" / "METEO" / "Resul_Meteo.dat")
    assert [float(r[header.index("Dir")]) for r in rows] == [315.0, 315.0]
    np.testing.assert_array_equal(case.meteo["wind_dir_from_deg"], 135.0)
    assert case.native["options"]["SIGMA_W_MIN"] == "0"
    for hour in ("00", "01"):
        header, rows = _table(DATA / "RESULT_SOUTHKENSINGTON" / "RUES_PAR_HEURE"
                              / f"Rues_20140107{hour}.dat")
        assert len(rows) == 46
        assert {r[header.index("Sigma_wH")] for r in rows} == {"0.30"}


def test_read_case_names_both_formats_for_an_unknown_path(tmp_path):
    with pytest.raises(ValueError, match="munich.cfg.*SIRANE master"):
        read_case(tmp_path / "nothing.txt")


def test_a_missing_data_file_names_the_path_and_the_convention(tmp_path):
    root = _deck_copy(tmp_path)
    (root / "FOND" / "Concentration_Fond.dat").unlink()
    with pytest.raises(FileNotFoundError, match="Concentration_Fond.dat.*master file"):
        read_case(root / MASTER.name)


@pytest.mark.parametrize("relative, old, new, error, match", [
    ("ESPECES/SrceGroup.dat", "Id\n", "Id\ntraffic\n", NotImplementedError,
     "SrceGroup.dat.*source group"),
    ("EMISSIONS/EMIS_LIN/Emis_Rues.dat", "45\t 0\t0\t0\t", "44\t 0\t0\t0\t", ValueError,
     "Emis_Rues.dat lists street 44 twice"),
    ("FOND/Concentration_Fond.dat", "07/01/2014 01:00\t0\t0\t0",
     "07/01/2014 00:00\t0\t0\t0", ValueError, "two rows dated 2014-01-07T00:00"),
    ("METEO/Site_Meteo.dat", "Albedo = 1", "Albedo du site = 1", ValueError,
     "Site_Meteo.dat.*Albedo du site"),
])
def test_deck_files_outside_what_the_reader_handles_are_named(tmp_path, relative, old, new,
                                                              error, match):
    root = _deck_copy(tmp_path)
    _replace(root / relative, old, new)
    with pytest.raises(error, match=match):
        read_case(root / MASTER.name)


def test_an_inactive_species_is_dropped_from_every_array(tmp_path):
    root = _deck_copy(tmp_path)
    _replace(root / "ESPECES" / "Especes.dat", "O3\t1\t48", "O3\t0\t48")
    case = read_case(root / MASTER.name)
    assert case.species == ["NO2", "NO"]
    assert case.emissions.shape == (2, 46, 2) and not case.emissions.any()
