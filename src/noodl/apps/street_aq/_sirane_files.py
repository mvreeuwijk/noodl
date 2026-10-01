"""SIRANE v2.1's own input-deck format: the master `Donnees_*.dat` file, its site files, the
Shapefile street network, and the tab-separated meteo, emission, species and background
tables it names.

Private to `noodl.apps.street_aq`: `case.py` is the format-neutral public surface
(`StreetCase`, `read_case`); this module knows nothing about that dataclass -- it returns
plain dicts/arrays in the neutral vocabulary (`wind_dir_from_deg`, `wind_speed`,
`temperature`; kg/s; kg/m3), and every SIRANE name, unit and convention is translated here.

The format was read from SIRANE v2.1 rev 128's own files (LMFA / Ecole Centrale de Lyon;
the model is described in Soulhac et al. 2011, Atmos. Environ. 45:7379) -- the South
Kensington deck in `tests/data/street/sirane_south_kensington` and the program's default
tables. What that deck does not exercise is refused by
name rather than guessed (see `read_sirane_case`). `write_sirane_case` writes decks in the
same format (the South Kensington deck's layout and French labels, every value from this
module's own tables), and `write_sweep_manifest` the parameter table of a sweep of them.

**Master and site files.** Each data line is `<description> = <value>`; `/` starts a
comment. SIRANE matches the DESCRIPTION -- the human-readable label of its default tables,
in French or in English -- not an internal keyword, so this module carries its own table of
the labels it needs in both languages (`_MASTER_LABELS`, `_SITE_LABELS`) mapped to SIRANE's
internal keyword (`MOT-CLE`), and everything downstream is keyed by that keyword. Matching
ignores case and runs of whitespace.

**Units.** Street emissions in `Emis_Rues.dat` are g/s per street (inferred: SIRANE's output log
reports the total of these values as "Total des emissions lineiques = ... g/s"; no file
states it), converted to kg/s here; background concentrations are micrograms/m3 (inferred by
analogy with SIRANE's concentration output, which is in micrograms/m3), converted to kg/m3;
the meteo temperature is degrees Celsius, converted to K. The meteo direction is degrees
clockwise from north, the direction the wind blows FROM (meteorological), which is already
the neutral convention.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from noodl.apps.street_aq import _shapefile
from noodl.apps.street_aq.network import Street, StreetNetwork

G_PER_KG = 1e3
"""SIRANE's street-emission unit (g/s, inferred -- see the module docstring) per kg."""

UG_PER_KG = 1e9
"""SIRANE's concentration unit (micrograms/m3) per kg."""

CELSIUS_TO_K = 273.15

_MASTER_LABELS: dict[str, tuple[str, str]] = {
    "FICH_DIR_INPUT": ("Repertoire des donnees d'entree", "Input data folder"),
    "TYPE_FICH_RESEAU": ("Type de fichier de reseau [0/1/2]", "Network file type [0/1/2]"),
    "FICH_RESEAU": ("Fichier de reseau", "Network of streets file"),
    "FICH_SRCE_GROUP": ("Fichier de groupes de sources", "Sources groups file"),
    "FICH_SOURCES_PONCT": ("Fichier de sources ponctuelles", "Point source file"),
    "FICH_ESPECES": ("Fichier des especes", "Pollutants file"),
    "FICH_METEO": ("Fichier meteo", "Meteorological file"),
    "FICH_EMIS_RUE": ("Fichier d'evolution des emissions lineiques et surfaciques",
                      "Emissions file"),
    "FICH_POLL_FOND": ("Fichier de pollution de fond", "Background concentration file"),
    "DATE_DEB": ("Date de debut", "Begin date"),
    "DATE_FIN": ("Date de fin", "End date"),
    "FICH_SITE_DISP": ("Fichier de site de dispersion", "Dispersion site data file"),
    "Z0D_BAT": ("Rugosite aerodynamique des batiments [m]", "Building surface roughness [m]"),
    "H_R": ("Hauteur de reflexion des bouffees [m]", "Canopy height [m]"),
    "TYPE_METEO": ("Conditions meteorologiques [0/1/2/3]",
                   "Meteorological conditions [0/1/2/3]"),
    "FICH_SITE_METEO": ("Fichier de site de mesures meteorologiques",
                        "Meteorological site data file"),
    "INPUT_METEO": ("Donnees meteorologiques fournies [0/1/2]",
                    "Meteorological data provided [0/1/2]"),
    "B_STREET_U_SIGMA_W": ("Vitesse moyenne et sigma_w des rues fournis [0/1]",
                           "Street mean velocity and sigma_w provided [0/1]"),
    "U_MIN": ("Vitesse du vent minimale [m/s]", "Minimum wind velocity [m/s]"),
    "SIGMA_V_MIN": ("Ecart-type de vitesse sigmav minimal [m/s]", "Minimum sigma_v [m/s]"),
    "SIGMA_W_MIN": ("Ecart-type de vitesse sigmaw minimal [m/s]", "Minimum sigma_w [m/s]"),
    "TYPE_DISP": ("Modele de diffusion [0/1/2]", "Diffusion model [0/1/2]"),
    "CHAPMAN": ("Activation du modele chimique de Chapman [0/1]",
                "Activation of the Chapman chemical model [0/1]"),
    "B_EMIS_NOEQNO2": ("Emissions de NO en equivalent NO2 [0/1]",
                       "NO emissions in equivalent NO2 [0/1]"),
    "I_N_MOD_LIN": ("Nombre de modulations lineiques", "Number of lineic modulations"),
    "FICH_RECEPT": ("Fichier de position des recepteurs ponctuels", "Point receptor file"),
    "AFFICH": ("Niveau d'affichage [0/1/2]", "Verbose level [0/1/2]"),
    "FICH_DIR_RESUL": ("Repertoire d'ecriture des resultats", "Results folder"),
    "FICH_GRD_MET": ("Fichier de description de la grille meteo",
                     "Meteorological grid definition file"),
    "FICH_GRD_SORTIE": ("Fichier de description de la grille de sortie",
                        "Output grid definition file"),
    "FICH_GRD_EMIS_SURF": ("Fichier de description de la grille d'emissions surfaciques",
                           "Surface emissions grid definition file"),
    "CALC_GRID": ("Calcul sur la grille [0/1/2]", "Grid results output [0/1/2]"),
    "B_ECRIRE_DEPOT": ("Ecriture du depot [0/1]", "Writing deposition [0/1]"),
    "CALC_RUES": ("Ecriture des resultats sur les rues [0/1]", "Streets results output [0/1]"),
    "B_CALC_INTERSECT": ("Calcul de la concentration aux intersections [0/1]",
                         "Street intersection concentration calculation [0/1]"),
    "ECRIRE_RESEAU": ("Ecriture du fichier de reseau de rues [0/1]",
                      "Streets network output [0/1]"),
    "FORMAT_RUES_SORTIE": ("Format du fichier de rues [0/1]", "Streets results format [0/1]"),
    "FORMAT_CHP_SORTIE": ("Format du fichier de champ de concentration [0/1/2/3/4]",
                          "Grid results format [0/1/2/3/4]"),
    "FORMAT_IMAGE_SORTIE": ("Ecriture du champ de concentration au format image [0/1/2]",
                            "Image results output [0/1/2]"),
    "FICH_COLORMAP_ESPECES": ("Fichier de colormap des especes", "Pollutants colormap file"),
    "FICH_COLORMAP_NB": ("Fichier de colormap des depassements", "Exceedence colormap file"),
    "B_CALC_STAT": ("Calcul des statistiques [0/1]", "Statistics calculation [0/1]"),
    "B_CONC_JOUR": ("Calcul des concentrations journalieres [0/1]",
                    "Daily averaged concentrations calculation [0/1]"),
    "FICH_PERCENT": ("Fichier de percentiles", "Percentiles file"),
    "FICH_SEUILS": ("Fichier de seuils de depassement", "Thresholds file"),
    "B_BOUFFEES": ("Prise en compte des bouffees [0/1]", "Puffs activation [0/1]"),
    "B_SRCE_PCT": ("Prise en compte des sources ponctuelles [0/1]",
                   "Point sources activation [0/1]"),
    "B_RETRO": ("Prise en compte des retrotrajectoires [0/1]",
                "Retrotrajectory activation [0/1]"),
    "B_PANACHE": ("Prise en compte des rues-panaches [0/1]", "Streets-plumes activation [0/1]"),
    "B_FOND": ("Prise en compte de la pollution de fond [0/1]",
               "Background concentration activation [0/1]"),
    "RATIO_GRILLE": ("Ratio entre pas meteo et pas retrotrajectoires",
                     "Ratio meteo and retrotrajectory grid"),
    "RETRO_BUFF": ("Zone tampon en cellules pour le calcul des retrotrajectoires",
                   "Buffer zone in cells for the retrotrajectory calculation"),
    "B_RUE_DECOUP": ("Decoupage des rues sur la grille retrotrajectoires [0/1]",
                     "Streets subdivision on the retrotrajectory grid [0/1]"),
    # Keys the South Kensington deck leaves at SIRANE's defaults.
    "FICH_RUE": ("Fichier de rue", "Streets file"),
    "FICH_NOEUD": ("Fichier de noeud", "Intersections file"),
    "KY": ("Diffusivite turbulente horizontale [m2/s]",
           "Horizontal turbulent diffusivity [m2/s]"),
    "KZ": ("Diffusivite turbulente verticale [m2/s]", "Vertical turbulent diffusivity [m2/s]"),
    "FICH_MASQUE_SORTIE": ("Fichier de masque de la grille de sortie", "Mask points file"),
    "B_GRID_BLANKED": ("Filtrage des points masques [0/1]", "Mask points filtering [0/1]"),
    "N_MAX_THREADS": ("Nombre maximum de threads utilises", "Maximum threads used"),
    "SEUIL_GAUSS": ("Seuil sur sigma pour negliger une bouffee",
                    "Sigma threshold to neglect a puff"),
    "SEUIL_DEBIT": ("Seuil sur le debit d'emission des bouffees",
                    "Puff emission rate threshold"),
    "SEUIL_PONCT": ("Seuil L/Sigma pour considerer une source comme ponctuelle",
                    "L/Sigma threshold to model a source as point"),
    "RATIO_BOUFFEE": ("Ratio du sous pas de temps pour le modele a bouffees",
                      "Ratio of subtime step for the puff model"),
    "SRCEPCT_BUFF": ("Zone tampon en cellules pour le calcul des sources ponctuelles",
                     "Buffer zone in cells for the point sources calculation"),
}
"""SIRANE master-file keyword -> (French label, English label): every run-control key of
SIRANE v2.1 (65), in its own two wordings (the descriptions of its `Don_Defaut_FR.dat` /
`Don_Defaut_EN.dat`). A master-file label matching neither wording of any key is refused by
name (`read_sirane_case`): SIRANE itself would not recognise it either."""

_SITE_LABELS: dict[str, tuple[str, str]] = {
    "LATITUDE": ("Latitude [deg]", "Latitude [deg]"),
    "ALTITUDE": ("Hauteur par rapport au sol [m]", "Height over ground [m]"),
    "Z0D": ("Rugosite aerodynamique [m]", "Aerodynamic roughness [m]"),
    "ZDISPL": ("Epaisseur de deplacement [m]", "Displacement height [m]"),
    "ALBEDO": ("Albedo", "Albedo"),
    "EMISSIVITE": ("Emissivite", "Emissivity"),
    "PRIESTLEY_TAYLOR": ("Coefficient de Priestley-Taylor", "Priestley-Taylor coefficient"),
}
"""SIRANE site-file keyword -> (French label, English label) (its `Site_Defaut_FR.dat` /
`Site_Defaut_EN.dat` descriptions). `ALTITUDE` is the height of the measurement above the
ground: for the meteo site, the height of the measured wind speed."""

_DEFAULTS: dict[str, str] = {
    "TYPE_FICH_RESEAU": "0",
    "TYPE_METEO": "0",
    "INPUT_METEO": "0",
    "B_STREET_U_SIGMA_W": "0",
    "B_EMIS_NOEQNO2": "0",
    "B_SRCE_PCT": "1",
    "I_N_MOD_LIN": "1",
    "Z0D_BAT": "0.05",
    "B_FOND": "1",
    "B_PANACHE": "1",
    "B_RETRO": "1",
    "TYPE_DISP": "2",
    "CHAPMAN": "0",
}
"""SIRANE's own default for each key this reader CONSULTS, used when the master file does
not set it (the defaults of SIRANE's `Don_Defaut_*.dat`). `native["options"]` records only
what the master file itself sets."""

_REQUIRED = ("DATE_DEB", "DATE_FIN", "FICH_RESEAU", "FICH_ESPECES", "FICH_METEO",
             "FICH_EMIS_RUE", "FICH_POLL_FOND", "FICH_SITE_DISP", "FICH_SITE_METEO")
"""Keys `read_sirane_case` cannot do without (no default makes sense for them)."""

_NETWORK_FIELDS = ("TYPE", "NDDEB", "NDFIN", "WG", "WD", "HG", "HD", "MODUL_EMIS")
"""The Shapefile DBF fields SIRANE itself consumes (its output log reports every other column
of the South Kensington network, e.g. `ID`, `LEN`, `X_S`, as "non utilisee")."""


def _normalise(label: str) -> str:
    return " ".join(label.split()).casefold()


def _lookup(table: dict[str, tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, labels in table.items():
        for label in labels:
            out[_normalise(label)] = key
    return out


_MASTER_LOOKUP = _lookup(_MASTER_LABELS)
_SITE_LOOKUP = _lookup(_SITE_LABELS)

_DATE_DAY_FIRST = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_DATE_YEAR_FIRST = re.compile(
    r"^(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def parse_date(text: str, *, where: str, who: str = "read_case") -> datetime:
    """A SIRANE date -> `datetime`. SIRANE's files mix three forms: day first with
    zero-padded fields (`07/01/2014 00:00:00` in the master file, `07/01/2014 00:00` in the
    meteo and background files), day first without padding (`7/1/2014 00:00`,
    `1/1/2014 0:00` in the emission-modulation and point-source files), and year first
    (`2014/01/07 00:00:00`, the per-street result files). Seconds are optional in all three.
    Anything else raises `ValueError` naming `where`. `who` names the calling reader in the
    message (`read_case` for a deck, `read_results` for a result directory)."""
    text = text.strip()
    match = _DATE_DAY_FIRST.match(text)
    if match:
        day, month, year = (int(g) for g in match.groups()[:3])
    else:
        match = _DATE_YEAR_FIRST.match(text)
        if not match:
            raise ValueError(
                f"{who}: {where}: {text!r} is not a SIRANE date (expected "
                f"DD/MM/YYYY HH:MM[:SS], D/M/YYYY H:MM or YYYY/MM/DD HH:MM[:SS])"
            )
        year, month, day = (int(g) for g in match.groups()[:3])
    hour, minute = int(match.group(4)), int(match.group(5))
    second = int(match.group(6) or 0)
    return datetime(year, month, day, hour, minute, second)


def _read_text(path: Path, *, what: str, who: str = "read_case") -> str:
    if not path.is_file():
        note = ("data files are resolved relative to the directory holding the master "
                "file" if who == "read_case" else
                "paths are resolved relative to the result directory")
        raise FileNotFoundError(f"{who}: {what} {path} does not exist ({note})")
    # SIRANE's own files are ISO-8859-1 (its French default table is); the labels used are
    # plain ASCII either way.
    return path.read_text(encoding="latin-1")


def _read_labelled(path: Path, lookup: dict[str, str], *, what: str) -> dict[str, str]:
    """Values by keyword from a `<label> = <value>` file. A label matching no keyword, or a
    keyword given twice with different values, raises `ValueError` naming it."""
    values: dict[str, str] = {}
    for raw in _read_text(path, what=what).splitlines():
        line = raw.strip()
        if not line or line.startswith("/") or "=" not in line:
            continue
        label, _, value = line.partition("=")
        label, value = " ".join(label.split()), value.strip()
        key = lookup.get(_normalise(label))
        if key is None:
            raise ValueError(
                f"read_case: {what} {path.name}: the label {label!r} is not one SIRANE v2.1 "
                f"defines (in French or English); check its spelling"
            )
        if key in values and values[key] != value:
            raise ValueError(
                f"read_case: {path.name} sets {key} ({label!r}) twice, to "
                f"{values[key]!r} and {value!r}"
            )
        values[key] = value
    return values


def _read_table(
    path: Path, *, what: str, who: str = "read_case"
) -> tuple[list[str], list[list[str]]]:
    """`(header, rows)` of a tab-separated SIRANE table. Fields are stripped; trailing empty
    fields (SIRANE's writers often end a row with a tab) are dropped; blank lines skipped.
    `who` names the calling reader in any error message (see `parse_date`)."""
    lines = [ln for ln in _read_text(path, what=what, who=who).splitlines() if ln.strip()]
    if not lines:
        return [], []

    def fields(line: str) -> list[str]:
        out = [f.strip() for f in line.split("\t")]
        while out and out[-1] == "":
            out.pop()
        return out

    header, rows = fields(lines[0]), [fields(ln) for ln in lines[1:]]
    for number, row in enumerate(rows, start=2):
        if len(row) < len(header):
            raise ValueError(
                f"{who}: {path.name} line {number} has {len(row)} field(s) {row}; its "
                f"header has {len(header)} ({header})"
            )
    return header, rows


def _column(header: list[str], name: str, path: Path, *, who: str = "read_case") -> int:
    try:
        return header.index(name)
    except ValueError:
        raise ValueError(
            f"{who}: {path.name} has no {name!r} column (columns: {header})"
        ) from None


def _rows_by_date(header: list[str], rows: list[list[str]], path: Path,
                  hours: list[datetime]) -> list[list[str]]:
    """The row of `rows` dated at each of `hours` (column `Date`), in that order. A missing
    hour or a duplicated date raises `ValueError` naming the file and the hour."""
    date_col = _column(header, "Date", path)
    by_date: dict[datetime, list[str]] = {}
    for row in rows:
        when = parse_date(row[date_col], where=path.name)
        if when in by_date:
            raise ValueError(f"read_case: {path.name} has two rows dated {when.isoformat()}")
        by_date[when] = row
    missing = [h.isoformat() for h in hours if h not in by_date]
    if missing:
        raise ValueError(
            f"read_case: {path.name} has no row for {missing}; it must cover every hour of "
            f"the period DATE_DEB..DATE_FIN"
        )
    return [by_date[h] for h in hours]


def _float(value: str, *, where: str, who: str = "read_case") -> float:
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{who}: {where}: {value!r} is not a number") from None


def _int_option(options: dict[str, str], key: str, path: Path) -> int:
    raw = options.get(key, _DEFAULTS[key])
    try:
        return int(float(raw))
    except ValueError:
        raise ValueError(f"read_case: {path.name}: {key} = {raw!r} is not an integer") from None


def _refuse_unless(options: dict[str, str], key: str, supported: int, path: Path,
                   why: str) -> None:
    value = _int_option(options, key, path)
    if value != supported:
        raise NotImplementedError(
            f"read_case: {path.name}: {key} = {value} is not supported; only {supported} "
            f"({why}) is"
        )


def _read_meteo_grid(root: Path, options: dict) -> dict[str, float] | None:
    """The meteo grid (`FICH_GRD_MET`, one row `Nx Ny xmin xmax ymin ymax`, `Nx`, `Ny`
    cells) with its cell sizes, or `None` when the deck names no such file or it is
    absent."""
    name = options.get("FICH_GRD_MET")
    if not name or not (root / name).is_file():
        return None
    path = root / name
    header, rows = _read_table(path, what="meteo grid file")
    keys = ["Nx", "Ny", "xmin", "xmax", "ymin", "ymax"]
    if [h.lower() for h in header[:6]] != [k.lower() for k in keys] or len(rows) != 1:
        raise ValueError(
            f"read_case: {path.name} must be the header {' '.join(keys)} and one row; got "
            f"header {header} and {len(rows)} row(s)"
        )
    v = {k: _float(x, where=f"{path.name} {k}") for k, x in zip(keys, rows[0][:6], strict=True)}
    if not (v["Nx"] >= 1 and v["Ny"] >= 1 and v["xmax"] > v["xmin"] and v["ymax"] > v["ymin"]):
        raise ValueError(f"read_case: {path.name} describes an empty grid: {v}")
    return {"nx": int(v["Nx"]), "ny": int(v["Ny"]), "xmin": v["xmin"], "xmax": v["xmax"],
            "ymin": v["ymin"], "ymax": v["ymax"], "dx": (v["xmax"] - v["xmin"]) / v["Nx"],
            "dy": (v["ymax"] - v["ymin"]) / v["Ny"]}


def _read_site(path: Path) -> dict[str, float]:
    values = _read_labelled(path, _SITE_LOOKUP, what="site file")
    return {key: _float(value, where=f"{path.name} {key}") for key, value in values.items()}


# ------------------------------------------------------------------------------- reading

def read_sirane_case(master: Path) -> dict:
    """Reads a SIRANE master file and the deck it names into a plain dict of the fields
    `StreetCase` takes: `network`, `times`, `start`, `street_ids`, `junction_ids`,
    `species`, `meteo`, `meteo_junction` (empty: SIRANE's meteorology is one station for the
    whole network), `emissions`, `background`, `native`.

    Data-file paths in the master file are resolved relative to the directory holding the
    master file. (SIRANE itself resolves them against `FICH_DIR_INPUT`, relative to its own
    working directory, which a reader cannot know; in SIRANE's decks the master file sits
    inside that input directory, so the two agree.)

    **Network** (`TYPE_FICH_RESEAU = 2`, a PolyLine Shapefile): street `i` is record `i`,
    named `str(i)` -- the record order is the street's id in every SIRANE output, not the
    DBF's own `ID` column, which SIRANE does not read. Its ends are the DBF's `NDDEB` ->
    `NDFIN` node ids, which are also the junction names; a junction's coordinates are the
    polyline endpoint carrying its id (two records disagreeing about one node's coordinates
    raise). Width `WG + WD` (the half-widths either side of the centreline); height
    `mean(HG, HD)`; length the endpoint distance; `z0_b` = `Z0D_BAT`, SIRANE's building
    surface roughness (inferred to be the wall roughness of Soulhac's in-canyon velocity
    profile -- the role `Street.z0_b` plays -- from its label; SIRANE's default 0.05 m when
    the master file does not set it). A street with `HG = 0` or `HD = 0` (buildings on one
    side only) is listed in `native["one_sided"]`; its height is still the mean, i.e. half
    the one side's.

    **Period**: hourly from `DATE_DEB` to `DATE_FIN` inclusive. **Species**: the rows of the
    species file with `Flag = 1`, in file order. **Meteo** (`TYPE_METEO = 0`,
    `INPUT_METEO = 0`: wind speed, direction, temperature, precipitation and cloud cover
    measured at the meteo site): `wind_speed` is the measured speed at the meteo site's
    `ALTITUDE`, `wind_dir_from_deg` the measured direction, `temperature` in K, each broadcast
    to `(n_hours, n_streets)`; precipitation (mm/h) and cloud cover (octas) are kept in
    `native["meteo_raw"]`. **Emissions**: each hour's street file (`Fich_Lin` of the
    emission-evolution file) times that hour's modulation `Mod_Lin_<g>_<species>` for the
    street's `MODUL_EMIS` group `g`, g/s -> kg/s. **Background**: micrograms/m3 -> kg/m3,
    the same at every street -- or zero when the master file switches the background off
    (`B_FOND = 0`).

    `native`: `options` (the master file's settings by SIRANE keyword, as the strings it
    gives), `physics` (the master switches that change SIRANE's results, with SIRANE's
    defaults applied: `background_on` = `B_FOND`, `plume_on` = `B_PANACHE` (the street-plume
    model above the roofs), `retro_on` = `B_RETRO` (the retrotrajectory model),
    `dispersion_model` = `TYPE_DISP` (int), `chemistry_on` = `CHAPMAN`; `read_case` does
    not act on the last four: the above-roof plume is `noodl.apps.street_aq.plume`, applied
    explicitly by `above_roof.street_steady_with_plume`), `site_disp`
    and `site_meteo` (the two site files by keyword, as floats), `meteo_grid` (the meteo
    grid file `FICH_GRD_MET`: `nx`, `ny`, `xmin`, `xmax`, `ymin`, `ymax` and the cell sizes
    `dx = (xmax - xmin) / nx`, `dy`, in m; `None` when the deck does not name the file or
    it is absent -- `plume` takes `dx` as `meteo_cell_dx`), `streets` (each record's
    SIRANE-consumed DBF fields, `_NETWORK_FIELDS`), `one_sided`, `meteo_raw`, `species_table`
    (the species file's rows for the active species).

    Refused by name, because the South Kensington deck does not exercise them and guessing
    their layout or meaning would be silent: a master or site label that SIRANE v2.1 does
    not define (e.g. a misspelt one), `TYPE_FICH_RESEAU` other than 2 (the plain-text
    network files), `TYPE_METEO` or `INPUT_METEO` other than 0, `B_STREET_U_SIGMA_W = 1`
    (user-supplied street velocities), `B_EMIS_NOEQNO2 = 1` (NO emitted as NO2-equivalent),
    a non-`NULL` `Fichier` column in the meteo file (inferred: a per-hour override file), a
    non-empty source-groups file, a surface-emission file with any data row, a table row
    with fewer fields than its header, and, while `B_SRCE_PCT = 1`, a point source whose own
    emission series is non-zero at any hour of the period, wherever it sits (noodl physics'
    street model has no point source).
    """
    master = Path(master)
    root = master.parent
    options = _read_labelled(master, _MASTER_LOOKUP, what="SIRANE master file")
    missing = [key for key in _REQUIRED if key not in options]
    if missing:
        raise ValueError(
            f"read_case: {master.name} sets no {missing} (labels "
            f"{[_MASTER_LABELS[k][0] for k in missing]} in French or "
            f"{[_MASTER_LABELS[k][1] for k in missing]} in English); is it a SIRANE master "
            f"file?"
        )
    _refuse_unless(options, "TYPE_FICH_RESEAU", 2, master, "the Shapefile network")
    _refuse_unless(options, "TYPE_METEO", 0, master, "meteorology measured at one site")
    _refuse_unless(options, "INPUT_METEO", 0, master, "wind speed and cloud cover given")
    _refuse_unless(options, "B_STREET_U_SIGMA_W", 0, master,
                   "SIRANE computes the street velocities itself")
    _refuse_unless(options, "B_EMIS_NOEQNO2", 0, master, "NO emitted as NO")

    start = parse_date(options["DATE_DEB"], where=f"{master.name} DATE_DEB")
    end = parse_date(options["DATE_FIN"], where=f"{master.name} DATE_FIN")
    if end < start:
        raise ValueError(
            f"read_case: {master.name}: DATE_FIN {end.isoformat()} is before DATE_DEB "
            f"{start.isoformat()}"
        )
    n_hours = int((end - start).total_seconds() // 3600) + 1
    hours = [start + timedelta(hours=k) for k in range(n_hours)]
    if hours[-1] != end:
        raise ValueError(
            f"read_case: {master.name}: DATE_DEB..DATE_FIN ({start.isoformat()} .. "
            f"{end.isoformat()}) is not a whole number of hours"
        )
    times = [3600.0 * k for k in range(n_hours)]

    # ------------------------------------------------------------------ network
    z0_b = _float(options.get("Z0D_BAT", _DEFAULTS["Z0D_BAT"]), where=f"{master.name} Z0D_BAT")
    lines, records = _shapefile.read_polylines(root / options["FICH_RESEAU"])
    x: dict[str, float] = {}
    y: dict[str, float] = {}
    streets: list[Street] = []
    street_fields: list[dict] = []
    one_sided: list[str] = []
    for i, (points, record) in enumerate(zip(lines, records, strict=True)):
        missing_fields = [f for f in _NETWORK_FIELDS if f not in record]
        if missing_fields:
            raise ValueError(
                f"read_case: {options['FICH_RESEAU']}.dbf has no {missing_fields} field(s); "
                f"a SIRANE network needs {list(_NETWORK_FIELDS)}"
            )
        u, v = str(record["NDDEB"]).strip(), str(record["NDFIN"]).strip()
        for node, (px, py) in ((u, points[0]), (v, points[-1])):
            if node in x and (x[node], y[node]) != (px, py):
                raise ValueError(
                    f"read_case: {options['FICH_RESEAU']}: node {node!r} is at "
                    f"({x[node]}, {y[node]}) in one record and ({px}, {py}) in record {i}"
                )
            x[node], y[node] = px, py
        hg, hd = float(record["HG"]), float(record["HD"])
        width = float(record["WG"]) + float(record["WD"])
        length = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
        name = str(i)
        streets.append(Street(name, u, v, length, width, 0.5 * (hg + hd), z0_b=z0_b))
        street_fields.append({f: record[f] for f in _NETWORK_FIELDS})
        if hg == 0.0 or hd == 0.0:
            one_sided.append(name)
    network = StreetNetwork(streets=streets, x=x, y=y)
    street_ids = [s.name for s in streets]
    n_streets = len(streets)

    # ------------------------------------------------------------------ species
    species_path = root / options["FICH_ESPECES"]
    header, rows = _read_table(species_path, what="species file")
    id_col, flag_col = _column(header, "Id", species_path), _column(header, "Flag", species_path)
    active = [row for row in rows if int(float(row[flag_col])) == 1]
    species = [row[id_col] for row in active]
    species_table = {row[id_col]: dict(zip(header, row, strict=False)) for row in active}
    if "FICH_SRCE_GROUP" in options:
        group_path = root / options["FICH_SRCE_GROUP"]
        _, group_rows = _read_table(group_path, what="source-groups file")
        if group_rows:
            raise NotImplementedError(
                f"read_case: {group_path.name} defines {len(group_rows)} source group(s); "
                f"this reader handles only an empty source-groups file"
            )

    # ------------------------------------------------------------------ sites
    site_disp = _read_site(root / options["FICH_SITE_DISP"])
    site_meteo = _read_site(root / options["FICH_SITE_METEO"])
    meteo_grid = _read_meteo_grid(root, options)

    # ------------------------------------------------------------------ meteo
    meteo_path = root / options["FICH_METEO"]
    header, rows = _read_table(meteo_path, what="meteo file")
    rows = _rows_by_date(header, rows, meteo_path, hours)
    if "Fichier" in header:
        col = header.index("Fichier")
        named = sorted({row[col] for row in rows if len(row) > col and row[col] != "NULL"})
        if named:
            raise NotImplementedError(
                f"read_case: {meteo_path.name}: the Fichier column names {named}; this "
                f"reader handles only NULL there (no per-hour meteo file)"
            )

    def meteo_column(name: str) -> np.ndarray:
        col = _column(header, name, meteo_path)
        return np.array([_float(row[col], where=f"{meteo_path.name} {name}") for row in rows],
                        dtype=np.float64)

    def per_street(series: np.ndarray) -> np.ndarray:
        return np.repeat(series[:, None], n_streets, axis=1)

    meteo = {
        "wind_dir_from_deg": per_street(meteo_column("Dir")),
        "wind_speed": per_street(meteo_column("U")),
        "temperature": per_street(meteo_column("Temp") + CELSIUS_TO_K),
    }
    meteo_raw = {"precip": meteo_column("Precip").tolist(),
                 "cloud": meteo_column("Cld").tolist()}

    # ------------------------------------------------------------------ emissions
    n_groups = _int_option(options, "I_N_MOD_LIN", master)
    groups = [int(float(f["MODUL_EMIS"])) for f in street_fields]
    bad = sorted({g for g in groups if not 0 <= g < n_groups})
    if bad:
        raise ValueError(
            f"read_case: street modulation group(s) {bad} (MODUL_EMIS) outside "
            f"0..{n_groups - 1} (I_N_MOD_LIN = {n_groups})"
        )
    evolution_path = root / options["FICH_EMIS_RUE"]
    header, rows = _read_table(evolution_path, what="emission-evolution file")
    rows = _rows_by_date(header, rows, evolution_path, hours)
    lin_col = _column(header, "Fich_Lin", evolution_path)
    surf_col = header.index("Fich_Surf") if "Fich_Surf" in header else None
    mod_cols = {(g, sp): _column(header, f"Mod_Lin_{g}_{sp}", evolution_path)
                for g in range(n_groups) for sp in species}
    street_files: dict[str, np.ndarray] = {}
    checked_surface: set[str] = set()
    emissions = np.zeros((n_hours, n_streets, len(species)), dtype=np.float64)
    group_index = np.asarray(groups, dtype=np.int64)
    for k, row in enumerate(rows):
        lin_name = row[lin_col]
        if lin_name not in street_files:
            street_files[lin_name] = _read_street_emissions(root / lin_name, species,
                                                            n_streets)
        modulation = np.array(
            [[_float(row[mod_cols[(g, sp)]], where=f"{evolution_path.name} Mod_Lin_{g}_{sp}")
              for sp in species] for g in range(n_groups)], dtype=np.float64)
        emissions[k] = street_files[lin_name] * modulation[group_index]
        if surf_col is not None and row[surf_col] not in checked_surface:
            _refuse_surface_emissions(root / row[surf_col])
            checked_surface.add(row[surf_col])
    emissions /= G_PER_KG

    # ------------------------------------------------------------------ point sources
    if "FICH_SOURCES_PONCT" in options and _int_option(options, "B_SRCE_PCT", master) == 1:
        _refuse_emitting_point_sources(root, root / options["FICH_SOURCES_PONCT"], hours)

    # ------------------------------------------------------------------ background
    background_path = root / options["FICH_POLL_FOND"]
    header, rows = _read_table(background_path, what="background file")
    rows = _rows_by_date(header, rows, background_path, hours)
    cols = [_column(header, sp, background_path) for sp in species]
    series = np.array([[_float(row[c], where=f"{background_path.name} {sp}")
                        for c, sp in zip(cols, species, strict=True)] for row in rows],
                      dtype=np.float64).reshape(n_hours, len(species))
    background = np.repeat(series[:, None, :], n_streets, axis=1) / UG_PER_KG
    physics = {
        "background_on": _int_option(options, "B_FOND", master) == 1,
        "plume_on": _int_option(options, "B_PANACHE", master) == 1,
        "retro_on": _int_option(options, "B_RETRO", master) == 1,
        "dispersion_model": _int_option(options, "TYPE_DISP", master),
        "chemistry_on": _int_option(options, "CHAPMAN", master) == 1,
    }
    if not physics["background_on"]:
        background = np.zeros_like(background)

    native = {
        "options": options,
        "physics": physics,
        "site_disp": site_disp,
        "site_meteo": site_meteo,
        "meteo_grid": meteo_grid,
        "streets": street_fields,
        "one_sided": one_sided,
        "meteo_raw": meteo_raw,
        "species_table": species_table,
    }
    return dict(
        network=network, times=times, start=start, street_ids=street_ids,
        junction_ids=list(network.junctions), species=species, meteo=meteo,
        meteo_junction={}, emissions=emissions, background=background, native=native,
    )


def _read_street_emissions(path: Path, species: list[str], n_streets: int) -> np.ndarray:
    """`(n_streets, n_species)` g/s from a street-emission file (`Id` = the street's record
    index, one column per species). Every street must appear exactly once."""
    header, rows = _read_table(path, what="street-emission file")
    id_col = _column(header, "Id", path)
    cols = [_column(header, sp, path) for sp in species]
    out = np.full((n_streets, len(species)), np.nan, dtype=np.float64)
    for row in rows:
        street = int(float(row[id_col]))
        if not 0 <= street < n_streets:
            raise ValueError(
                f"read_case: {path.name} has street Id {street}; the network has streets "
                f"0..{n_streets - 1}"
            )
        if not np.isnan(out[street]).all():
            raise ValueError(f"read_case: {path.name} lists street {street} twice")
        out[street] = [_float(row[c], where=f"{path.name} street {street} {sp}")
                       for c, sp in zip(cols, species, strict=True)]
    absent = [i for i in range(n_streets) if np.isnan(out[i]).any()]
    if absent:
        raise ValueError(f"read_case: {path.name} has no row for street(s) {absent}")
    return out


def _refuse_surface_emissions(path: Path) -> None:
    _, rows = _read_table(path, what="surface-emission file")
    if rows:
        raise NotImplementedError(
            f"read_case: {path.name} has {len(rows)} row(s) of surface emissions; noodl "
            f"physics' street model has no surface (gridded) source, so only an empty "
            f"surface-emission file is read"
        )


# ------------------------------------------------------------------------------- results

_RESULT_HOUR_RE = re.compile(r"^Rues_(\d{4})(\d{2})(\d{2})(\d{2})\.dat$")
"""`RUES_PAR_HEURE/Rues_<YYYYMMDDHH>.dat`'s own hour, embedded in its name rather than
inside the file (see `read_sirane_results`)."""


def _hour_from_result_filename(name: str) -> datetime:
    match = _RESULT_HOUR_RE.match(name)
    if not match:
        raise ValueError(
            f"read_results: {name!r} is not a SIRANE hourly-street-results file name "
            f"(expected Rues_<YYYYMMDDHH>.dat)"
        )
    year, month, day, hour = (int(g) for g in match.groups())
    return datetime(year, month, day, hour)


def _no_negative_zero(value: float) -> float:
    """SIRANE prints `-0.00` for a value that rounds to zero from below; `-0.0 + 0.0 == 0.0`
    (IEEE 754) turns the parsed float's sign bit off, so a reader never sees `-0.0`."""
    return value + 0.0


def read_sirane_results(result_dir: Path, *, case=None, hours: str = "case") -> dict:
    """Reads a SIRANE result directory into the fields `StreetResults` takes: `times`,
    `street_ids`, `species`, `c_in`, `c_above`, `u_canyon`, `sigma_w_roof`, `u_exchange`,
    `meteo`.

    **`RUES_PAR_HEURE/Rues_<YYYYMMDDHH>.dat`** -- one file per output hour, every street's
    row: `species` from its `Cint_<sp>` columns, in file order; `street_ids` from each row's
    `Id` (`"<a>_<b>"`, the street's record index `a` -- see the module docstring's network
    section for why `a` is authoritative, not the DBF's own `ID`). `Cint_<sp>` -> `c_in`,
    `Cext_<sp>` -> `c_above` (micrograms/m3 -> kg/m3); `U_moy` -> `u_canyon`, `Sigma_wH` ->
    `sigma_w_roof`, `u_d` -> `u_exchange` (already SI, m/s) -- `u_exchange` is read AS
    PRINTED, not recomputed from `sigma_w_roof`, so it can be checked against SIRANE's own
    closure (`EXCHANGE_SIGMA_W_RATIO * sigma_w_roof`, `noodl.apps.street_aq.canyon`).

    **`METEO/Resul_Meteo.dat`** -- the meteorological preprocessor's own derived state, one
    row per hour, dated by its `JJ/MM/AAAA` + `HH:MM` columns together: `Ustar` -> `u_star`,
    `SigmaTheta` -> `sigma_theta` (degrees -> radians -- SIRANE's own file, unlike every
    other angle this reader touches, is NOT already the neutral convention), `Hcla` ->
    `h_abl`, `Lmo` -> `lmo`, `U` -> `wind_speed`, `Dir` -> `wind_dir_from_deg`, `T` ->
    `temperature` (Celsius -> K). Every `meteo` value is broadcast to `(n_hours, n_streets)`,
    network-wide: SIRANE's meteorology is one station for the whole case. In `hours="case"`
    mode a wanted hour absent from `Resul_Meteo.dat` raises (that file is the current run's
    own, so a gap means the result directory is broken); in `hours="all"` mode -- where a
    wanted hour may be a stale one `RUES_PAR_HEURE` kept but `Resul_Meteo.dat` itself was
    overwritten without -- it reads as `nan` instead.

    A value SIRANE prints as `-0.00` reads as `0.0`, not `-0.0` (`_no_negative_zero`).

    `hours`: `"case"` (default) keeps only the hours inside `case`'s own period (`case.start`
    + `case.times`) -- `case` is required for this mode (a `ValueError` names the
    alternative when it is missing), since an archived result directory can mix hours from
    more than one run (see `tests/data/street/sirane_south_kensington`'s NOTICE.md): a case
    hour missing from the directory raises `ValueError` naming it. `"all"` returns every
    hour `RUES_PAR_HEURE` holds, in time order, with or without a `case`.

    A street-results file whose species or street order differs from another hour's, a
    `RUES_PAR_HEURE` with no file, or a `METEO/Resul_Meteo.dat` missing a wanted hour, each
    raise `ValueError` naming the files.
    """
    result_dir = Path(result_dir)
    hourly_dir = result_dir / "RUES_PAR_HEURE"
    available = {_hour_from_result_filename(p.name): p
                 for p in sorted(hourly_dir.glob("Rues_*.dat"))}
    if not available:
        raise FileNotFoundError(
            f"read_results: {hourly_dir} has no Rues_<YYYYMMDDHH>.dat file"
        )

    if hours == "all":
        wanted = sorted(available)
    elif hours == "case":
        if case is None:
            raise ValueError(
                "read_results: hours='case' needs case (to know the deck's own period); "
                "pass hours='all' to read every hour the result directory holds instead"
            )
        if case.start is None:
            raise ValueError(
                "read_results: case.start is None; hours='case' needs the case's own "
                "absolute period"
            )
        wanted = [case.start + timedelta(seconds=t) for t in case.times]
        missing = [h.isoformat() for h in wanted if h not in available]
        if missing:
            raise ValueError(
                f"read_results: {hourly_dir} has no file for hour(s) {missing} (the case's "
                f"own period) -- it may hold results from a different run (see NOTICE.md); "
                f"pass hours='all' to read every hour it has instead"
            )
    else:
        raise ValueError(f"read_results: hours must be 'case' or 'all', got {hours!r}")

    species: list[str] | None = None
    street_ids: list[str] | None = None
    hourly_headers: list[list[str]] = []
    hourly_rows: list[list[list[str]]] = []
    for hour in wanted:
        path = available[hour]
        header, rows = _read_table(path, what="hourly street-results file",
                                   who="read_results")
        this_species = [name.split("_", 1)[1] for name in header if name.startswith("Cint_")]
        if not this_species:
            raise ValueError(f"read_results: {path.name} has no Cint_<species> column")
        id_col = _column(header, "Id", path, who="read_results")
        this_ids = [row[id_col].split("_", 1)[0] for row in rows]
        if species is None:
            species, street_ids = this_species, this_ids
        elif this_species != species:
            raise ValueError(
                f"read_results: {path.name} lists species {this_species}, but "
                f"{available[wanted[0]].name} lists {species}"
            )
        elif this_ids != street_ids:
            raise ValueError(
                f"read_results: {path.name}'s streets are not in the same order as "
                f"{available[wanted[0]].name}"
            )
        hourly_headers.append(header)
        hourly_rows.append(rows)

    n_hours, n_streets = len(wanted), len(street_ids)

    def value(row: list[str], header: list[str], name: str, path: Path) -> float:
        col = _column(header, name, path, who="read_results")
        return _no_negative_zero(
            _float(row[col], where=f"{path.name} {name}", who="read_results")
        )

    u_canyon = np.zeros((n_hours, n_streets))
    sigma_w_roof = np.zeros((n_hours, n_streets))
    u_exchange = np.zeros((n_hours, n_streets))
    c_in = {sp: np.zeros((n_hours, n_streets)) for sp in species}
    c_above = {sp: np.zeros((n_hours, n_streets)) for sp in species}
    for h, (hour, header, rows) in enumerate(
        zip(wanted, hourly_headers, hourly_rows, strict=True)
    ):
        path = available[hour]
        if len(rows) != n_streets:
            raise ValueError(
                f"read_results: {path.name} has {len(rows)} street row(s), expected "
                f"{n_streets}"
            )
        for s, row in enumerate(rows):
            u_canyon[h, s] = value(row, header, "U_moy", path)
            sigma_w_roof[h, s] = value(row, header, "Sigma_wH", path)
            u_exchange[h, s] = value(row, header, "u_d", path)
            for sp in species:
                c_in[sp][h, s] = value(row, header, f"Cint_{sp}", path) / UG_PER_KG
                c_above[sp][h, s] = value(row, header, f"Cext_{sp}", path) / UG_PER_KG

    meteo_path = result_dir / "METEO" / "Resul_Meteo.dat"
    header, rows = _read_table(meteo_path, what="processed meteorology file",
                               who="read_results")
    date_col = _column(header, "JJ/MM/AAAA", meteo_path, who="read_results")
    time_col = _column(header, "HH:MM", meteo_path, who="read_results")
    by_hour: dict[datetime, list[str]] = {}
    for row in rows:
        when = parse_date(f"{row[date_col]} {row[time_col]}", where=meteo_path.name,
                          who="read_results")
        by_hour[when] = row
    missing = [h.isoformat() for h in wanted if h not in by_hour]
    if missing and hours == "case":
        # In "case" mode every wanted hour is the run that produced RUES_PAR_HEURE's own
        # files, so its processed meteorology should be there too; a gap means the result
        # directory is broken, not just archived-over (see the "all" mode note below).
        raise ValueError(f"read_results: {meteo_path.name} has no row for hour(s) {missing}")

    def meteo_column(name: str) -> np.ndarray:
        # In "all" mode a stale hour (see the docstring) commonly has no processed-meteo
        # row at all -- SIRANE overwrites Resul_Meteo.dat with only its CURRENT run's hours,
        # unlike RUES_PAR_HEURE, whose per-hour files simply accumulate -- so it reads as
        # NaN rather than raising.
        col = _column(header, name, meteo_path, who="read_results")
        series = np.array([
            _no_negative_zero(_float(by_hour[h][col], where=f"{meteo_path.name} {name}",
                                     who="read_results"))
            if h in by_hour else math.nan
            for h in wanted
        ])
        return np.repeat(series[:, None], n_streets, axis=1)

    meteo = {
        "u_star": meteo_column("Ustar"),
        "sigma_theta": np.radians(meteo_column("SigmaTheta")),
        "h_abl": meteo_column("Hcla"),
        "lmo": meteo_column("Lmo"),
        "wind_speed": meteo_column("U"),
        "wind_dir_from_deg": meteo_column("Dir"),
        "temperature": meteo_column("T") + CELSIUS_TO_K,
    }

    return dict(
        times=wanted, street_ids=street_ids, species=species, c_in=c_in, c_above=c_above,
        u_canyon=u_canyon, sigma_w_roof=sigma_w_roof, u_exchange=u_exchange, meteo=meteo,
    )


def _refuse_emitting_point_sources(root: Path, path: Path, hours: list[datetime]) -> None:
    """Refuses any point source whose own emission series (its `Fichier`, relative to
    `root`) is non-zero at any hour of the period, wherever the source sits: noodl physics'
    street model has no point source. A source with no series (`NULL`) is refused too."""
    header, rows = _read_table(path, what="point-source file")
    id_col, file_col = _column(header, "Id", path), _column(header, "Fichier", path)
    for row in rows:
        name = row[id_col]
        if row[file_col] == "NULL":
            raise NotImplementedError(
                f"read_case: {path.name}: point source {name!r} has no emission series "
                f"(Fichier NULL); this reader needs each source's series to check it is zero"
            )
        series_path = root / row[file_col]
        s_header, s_rows = _read_table(series_path, what="point-source emission series")
        s_rows = _rows_by_date(s_header, s_rows, series_path, hours)
        date_col = s_header.index("Date")
        for s_row, hour in zip(s_rows, hours, strict=True):
            values = [_float(v, where=f"{series_path.name} {hour.isoformat()}")
                      for c, v in enumerate(s_row) if c != date_col]
            if any(v != 0.0 for v in values):
                raise NotImplementedError(
                    f"read_case: {path.name}: point source {name!r} emits at "
                    f"{hour.isoformat()} ({series_path.name}); noodl physics' street model "
                    f"has no point source, so only point sources with zero emission over "
                    f"the period are read (and ignored)"
                )


# ------------------------------------------------------------------------------- writing

MASTER_FILE = "Donnees.dat"
"""The master file `write_sirane_case` writes at the top of its deck directory."""

RESULT_SUBDIR = "RESULT"
"""The results folder `write_sirane_case` names by default, inside the deck directory, so
that every deck of a sweep keeps its own results."""

_DECK_FILES = {
    "FICH_RESEAU": "RESEAU/Reseau",
    "FICH_SITE_DISP": "RESEAU/Site_Disp.dat",
    "FICH_METEO": "METEO/Meteo_change_1h.dat",
    "FICH_SITE_METEO": "METEO/Site_Meteo.dat",
    "FICH_EMIS_RUE": "EMISSIONS/Emissions_Lin_Surf.dat",
    "FICH_SOURCES_PONCT": "EMISSIONS/Sources_Point.dat",
    "FICH_ESPECES": "ESPECES/Especes.dat",
    "FICH_SRCE_GROUP": "ESPECES/SrceGroup.dat",
    "FICH_POLL_FOND": "FOND/Concentration_Fond.dat",
    "FICH_RECEPT": "RECEPTEURS/recep.dat",
    "FICH_GRD_MET": "GRILLES/Grille_Meteo.dat",
    "FICH_GRD_SORTIE": "GRILLES/Grille_Sortie.dat",
    "FICH_GRD_EMIS_SURF": "GRILLES/Grille_Emis_Surf.dat",
    "FICH_PERCENT": "STATISTIQUES/Percentiles.dat",
    "FICH_SEUILS": "STATISTIQUES/Seuils.dat",
}
"""Master-file keyword -> the deck file `write_sirane_case` writes for it, relative to the
input folder (`FICH_DIR_INPUT`). The layout and names follow the South Kensington deck's."""

_STREET_EMISSION_DIR = "EMISSIONS/EMIS_LIN"
_SURFACE_EMISSION_FILE = "EMISSIONS/EMIS_SURF/EmisSurf_nulle.dat"
_POINT_SERIES_FILE = "EMISSIONS/EMIS_PONCT/Emis_ponct_nulle.dat"

_WRITE_DEFAULTS: dict[str, str] = {
    # Fixed by what this writer writes (and what `read_sirane_case` reads back).
    "TYPE_FICH_RESEAU": "2",
    "TYPE_METEO": "0",
    "B_STREET_U_SIGMA_W": "0",
    "B_EMIS_NOEQNO2": "0",
    "I_N_MOD_LIN": "1",
    "B_RUE_DECOUP": "0",
    # Physics and numerics: a case read from a SIRANE deck carries its own over these
    # (`_CARRIED_OVER`), and `options` overrides both.
    "CHAPMAN": "0",
    "B_PANACHE": "1",
    "B_FOND": "1",
    "H_R": "20.0",
    "U_MIN": "0.2",
    "SIGMA_V_MIN": "0.5",
    "SIGMA_W_MIN": "0.3",
    "RATIO_GRILLE": "1",
    "RETRO_BUFF": "0",
    # Output: street results only, as tab-separated text; no grid, no images.
    "AFFICH": "2",
    "B_CALC_STAT": "1",
    "B_CONC_JOUR": "0",
    "CALC_GRID": "0",
    "B_ECRIRE_DEPOT": "0",
    "CALC_RUES": "1",
    "FORMAT_RUES_SORTIE": "0",
    "B_CALC_INTERSECT": "0",
    "FORMAT_CHP_SORTIE": "4",
    "FORMAT_IMAGE_SORTIE": "0",
}
"""The master-file values `write_sirane_case` writes, by SIRANE keyword (noodl physics' own
choices, stated here rather than taken from SIRANE's default tables). `B_RUE_DECOUP = 0`
keeps one result row per street (`Id` `"<i>_<i>"`, which `read_sirane_results` relies on).
`H_R`, `RATIO_GRILLE = 1` and `RETRO_BUFF = 0` are the South Kensington deck's settings;
`U_MIN`/`SIGMA_V_MIN`/`SIGMA_W_MIN` = 0.2/0.5/0.3 m/s are the floors of the archived South
Kensington SIRANE run (the South Kensington fixture deck zeroes them). `CALC_GRID = 0` skips
the concentration grid (so no colormap or image file is needed, and a run is much faster);
`FORMAT_CHP_SORTIE` is mandatory in SIRANE's table even then. `B_PANACHE` and `B_FOND`
are written only when 0 (1 is SIRANE's own default), so a default deck uses only labels the
South Kensington deck itself sets."""

_CARRIED_OVER = ("CHAPMAN", "B_PANACHE", "B_FOND", "B_RETRO", "B_BOUFFEES", "B_SRCE_PCT",
                 "TYPE_DISP", "KY", "KZ", "H_R", "U_MIN", "SIGMA_V_MIN", "SIGMA_W_MIN",
                 "SEUIL_GAUSS", "SEUIL_DEBIT", "SEUIL_PONCT", "RATIO_GRILLE", "RATIO_BOUFFEE",
                 "RETRO_BUFF", "SRCEPCT_BUFF", "N_MAX_THREADS", "AFFICH")
"""The master-file keywords a case read from a SIRANE deck writes back (its physics and
numerical settings; never a file name, a folder, an output switch, or a setting this writer
fixes). Each is also accepted as a `write_sirane_case` option, spelt as the keyword."""

_SHORT_OPTIONS = {"chapman": "CHAPMAN", "plume": "B_PANACHE"}
"""`write_sirane_case` option -> the 0/1 master-file keyword it sets."""

_RANGES: dict[str, tuple[float, float, bool]] = {
    # master file (SIRANE's Don_Defaut_*.dat min/max; True = integer)
    "CHAPMAN": (0, 1, True), "B_PANACHE": (0, 1, True), "B_FOND": (0, 1, True),
    "B_RETRO": (0, 1, True), "B_BOUFFEES": (0, 1, True), "B_SRCE_PCT": (0, 1, True),
    "TYPE_DISP": (0, 2, True), "KY": (0.0, 100.0, False), "KZ": (0.0, 100.0, False),
    "H_R": (0.0, 100.0, False), "Z0D_BAT": (0.0, 1.0, False),
    "U_MIN": (0.0, 5.0, False), "SIGMA_V_MIN": (0.0, 2.0, False),
    "SIGMA_W_MIN": (0.0, 2.0, False), "SEUIL_GAUSS": (0.0, 100000.0, False),
    "SEUIL_DEBIT": (0.0, 10000.0, False), "SEUIL_PONCT": (0.0, 10000.0, False),
    "RATIO_GRILLE": (1, 6, True), "RATIO_BOUFFEE": (1, 60, True), "RETRO_BUFF": (0, 10, True),
    "SRCEPCT_BUFF": (0, 1000, True), "N_MAX_THREADS": (1, 1000, True), "AFFICH": (0, 2, True),
    # site files (SIRANE's Site_Defaut_*.dat min/max)
    "LATITUDE": (-90.0, 90.0, False), "ALTITUDE": (0.0, 90.0, False),
    "Z0D": (0.0, 3.0, False), "ZDISPL": (0.0, 50.0, False), "ALBEDO": (0.0, 1.0, False),
    "EMISSIVITE": (0.0, 1.0, False), "PRIESTLEY_TAYLOR": (0.0, 1.0, False),
}
"""SIRANE's accepted range of each numeric value `write_sirane_case` writes, by keyword:
`(min, max, integer)`, as SIRANE v2.1's default tables state them. A value outside its
range (or a non-integer for an integer key) is refused by name before anything is written,
rather than written as a deck SIRANE would reject."""

RESULT_SUBFOLDERS = ("METEO", "RECEPT", "RECEPT_STAT", "RUES_PAR_HEURE", "RUES_PAR_RUE",
                     "RUES_STAT", "GRILLE", "GRILLE_STAT", "IMAGES", "IMAGES_STAT")
"""The folders SIRANE writes its results into, under `FICH_DIR_RESUL` (as its output log lists
them). `write_sirane_case` creates them all in advance: the only output log available shows
SIRANE finding them already there ("existe deja"), never creating them."""


def _check_range(key: str, value, *, where: str, who: str = "write_case") -> None:
    low, high, integer = _RANGES[key]
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{who}: {key} ({where}) = {value!r} is not a number") from None
    if integer and number != int(number):
        raise ValueError(f"{who}: {key} ({where}) = {value!r} must be a whole number")
    if not low <= number <= high:
        raise ValueError(
            f"{who}: {key} ({where}) = {value!r} is outside SIRANE's range {low}..{high}"
        )


_OPTION_RANGE_KEY = {"latitude": "LATITUDE", "measurement_height": "ALTITUDE",
                     "deposition": "CHAPMAN"}
"""Short options range-checked as a keyword of `_RANGES` (`deposition` is 0/1, as
`CHAPMAN` is)."""


def check_sirane_options(options, species, *, who: str = "write_case") -> dict:
    """`options` for `write_sirane_case`, checked and normalised (the short names `chapman`
    and `plume` become their keywords) -- without writing anything, so a sweep can check
    every variant before writing its first deck. An unknown option, a value outside
    SIRANE's range (`_RANGES`), or chemistry asked for without NO2, NO and O3 among
    `species` raises `ValueError` naming it."""
    options = dict(options or {})
    known = ({"deposition", "latitude", "measurement_height", "input_dir", "result_dir"}
             | set(_SHORT_OPTIONS) | set(_CARRIED_OVER))
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(
            f"{who}: format='sirane' options {unknown} are not understood; accepted: "
            f"{sorted(known)}"
        )
    for key, value in options.items():
        range_key = _OPTION_RANGE_KEY.get(key, _SHORT_OPTIONS.get(key, key))
        if range_key in _RANGES:
            _check_range(range_key, value, where=f"option {key!r}", who=who)
    for short, key in _SHORT_OPTIONS.items():
        if short in options:
            if key in options:
                raise ValueError(f"{who}: options give both {short!r} and {key!r}")
            options[key] = options.pop(short)
    if "CHAPMAN" in options and int(float(options["CHAPMAN"])) == 1:
        missing = [sp for sp in _CHAPMAN_SPECIES if sp not in species]
        if missing:
            raise ValueError(
                f"{who}: chemistry (CHAPMAN = 1) acts on NO2, NO and O3; the species "
                f"{list(species)} lack {missing}"
            )
    for key in ("deposition", *_CARRIED_OVER):
        if key in options and _RANGES.get(_OPTION_RANGE_KEY.get(key, key), (0, 0, False))[2]:
            options[key] = str(int(float(options[key])))
    return options

_SITE_DEFAULTS: dict[str, float] = {
    "LATITUDE": 51.5,
    "ALTITUDE": 10.0,
    "Z0D": 1.0,
    "ALBEDO": 0.2,
    "EMISSIVITE": 0.88,
    "PRIESTLEY_TAYLOR": 0.5,
}
"""The site-file values `write_sirane_case` writes for a case not read from a SIRANE deck:
latitude 51.5 deg (London), measurement height 10 m, roughness 1.0 m (the South Kensington
deck's), albedo 0.2, emissivity 0.88, Priestley-Taylor coefficient 0.5. The displacement
height differs between the two files (`_DISPLACEMENT`)."""

_DISPLACEMENT = {"site_meteo": 0.0, "site_disp": 13.0}
"""`ZDISPL` (m) of the meteo site (a mast in the open) and of the dispersion site (the urban
canopy), as in the South Kensington deck."""

_METEO_DEFAULTS = {"temperature_c": 4.0, "precip": 0.0, "cloud": 5}
"""Meteo columns a case may not carry: temperature (Celsius), precipitation (mm/h) and
cloud cover (octas) -- the South Kensington deck's values (4 C, dry, 5 octas)."""

_SPECIES_PROPERTIES: dict[str, tuple[float, float, float]] = {
    "NO2": (46.0, 0.0, 0.0),
    "NO": (30.0, 0.0, 0.0),
    "O3": (48.0, 0.0, 0.0),
    "PM": (100.0, 1.0e-5, 1000.0),
    "PM25": (100.0, 2.5e-6, 1000.0),
    "CO": (28.0, 0.0, 0.0),
    "C6H6": (78.0, 0.0, 0.0),
}
"""The species of the South Kensington deck's species file -> (molar mass g/mol, particle
diameter m, particle density kg/m3). `write_sirane_case` writes only these names: a species
SIRANE may not know is refused by name rather than invented. A passive tracer is written as
one of them with chemistry (`CHAPMAN`) and deposition off."""

_CHAPMAN_SPECIES = ("NO2", "NO", "O3")

_DBF_FIELDS = [("TYPE", "N", 9, 0), ("NDDEB", "C", 20, 0), ("NDFIN", "C", 20, 0),
               ("WG", "N", 19, 9), ("WD", "N", 19, 9), ("HG", "N", 19, 9),
               ("HD", "N", 19, 9), ("MODUL_EMIS", "N", 9, 0)]
"""The network DBF fields `write_sirane_case` writes: exactly the ones SIRANE consumes
(`_NETWORK_FIELDS`). Integers are 9 wide (read as integers by any dBase reader); lengths
carry 9 decimals, so a width or height re-reads to within 1e-9 m."""

_GRID_MARGIN = 50
"""Metres the output and surface-emission grids extend beyond the outermost junctions (the
South Kensington deck's extend about 40 m), so that every street lies inside SIRANE's
domain; the meteo grid extends a further 50 m, rounded out to whole 100 m."""

_CRLF = "\r\n"


def _num(value: float) -> str:
    """A number as SIRANE's tables print it: a whole number without a decimal point
    (`0`, `1`, `-16000`), anything else as Python's shortest round-tripping repr."""
    value = float(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _date_master(when: datetime) -> str:
    return when.strftime("%d/%m/%Y %H:%M:%S")


def _date_table(when: datetime) -> str:
    return when.strftime("%d/%m/%Y %H:%M")


def _date_unpadded(when: datetime) -> str:
    return f"{when.day}/{when.month}/{when.year} {when.strftime('%H:%M')}"


def _write_lines(path: Path, lines: list[str], *, final_newline: bool = True) -> None:
    """`lines` joined by CRLF (SIRANE is a Windows program; its own deck is CRLF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _CRLF.join(lines) + (_CRLF if final_newline else "")
    path.write_bytes(text.encode("latin-1"))


def _labelled(title: str, values: list[tuple[str, str]],
              labels: dict[str, tuple[str, str]]) -> list[str]:
    """A `/`-comment section title, then one `<French label> = <value>` line per key."""
    lines = ["/" + "-" * 30, f"/ {title}", "/" + "-" * 30]
    lines += [f"{labels[key][0]} = {value}" for key, value in values]
    return lines


def _whole(value: float, *, step: float) -> bool:
    return abs(value / step - round(value / step)) < 1e-9


def _uniform(label: str, values: np.ndarray, *, why: str) -> np.ndarray:
    """`values` `(n_hours, n_streets, ...)` -> its per-hour value `(n_hours, ...)`, refusing
    one that varies across streets (SIRANE has one value for the whole network)."""
    first = values[:, 0]
    if not np.allclose(values, first[:, None], rtol=1e-12, atol=0.0):
        raise ValueError(
            f"write_case: {label} varies between streets within an hour; SIRANE takes "
            f"{why} for the whole network"
        )
    return first


def write_sirane_case(
    out_dir: Path,
    *,
    network: StreetNetwork,
    times: list[float],
    start: datetime,
    junction_ids: list[str],
    species: list[str],
    meteo: dict[str, np.ndarray],
    emissions: np.ndarray,
    background: np.ndarray,
    native: dict | None = None,
    options: dict | None = None,
) -> Path:
    """Writes a complete SIRANE v2.1 deck under `out_dir` and returns the master file's path
    (`out_dir / MASTER_FILE`). Every input is in the neutral vocabulary (`StreetCase`'s);
    `native`, when given, is the `native` of a case read from a SIRANE deck
    (`read_sirane_case`).

    **Paths.** SIRANE resolves `FICH_DIR_INPUT` and `FICH_DIR_RESUL` against its own
    working directory, and every data file against `FICH_DIR_INPUT`. By default the input
    folder is `out_dir`'s own name and the results go to `<that name>/RESULT`, i.e.
    SIRANE's working directory is `out_dir`'s parent.
    Options `input_dir` and `result_dir` (paths relative to SIRANE's working directory,
    `/`-separated) change both; `input_dir` must be the trailing part of `out_dir` (SIRANE's
    working directory is what precedes it). The results folder and every subfolder SIRANE
    writes into (`RESULT_SUBFOLDERS`) are created in advance.

    Every numeric setting written -- options, a SIRANE case's carried-over settings and
    site values, `Z0D_BAT` -- is checked against SIRANE's own range (`_RANGES`) first, and
    refused by name outside it.

    **Master file**: French labels (as the South Kensington deck), the values of
    `_WRITE_DEFAULTS`, the period `DATE_DEB`..`DATE_FIN` from `start` and `times` (which
    must be whole consecutive hours from 0), `Z0D_BAT` = the streets' `z0_b` (one value for
    the network, within SIRANE's range 0..1 m).

    **Network**: a PolyLine Shapefile (`TYPE_FICH_RESEAU = 2`), one straight two-point
    record per street in network order (so street `i` is SIRANE's street `i`), from the
    `u` to the `v` junction's coordinates; a street whose `length` is not that distance is
    refused (SIRANE takes the length from the geometry). DBF fields `_DBF_FIELDS`: node ids
    `NDDEB`/`NDFIN` are `junction_ids` when every one is a distinct whole number (a case read
    from SIRANE keeps its own), else the junctions numbered 1, 2, ... in network order;
    `WG = WD = width / 2`, `HG = HD = height`, `TYPE = 0` -- except for a case read from
    SIRANE whose own fields still give each street's width (`WG + WD`) and height
    (`mean(HG, HD)`), which are written back as read (one-sided streets included);
    `MODUL_EMIS = 0` (one modulation group).

    **Site files**: `_SITE_DEFAULTS` with `_DISPLACEMENT`, or a SIRANE case's own; options
    `latitude` (deg) and `measurement_height` (the meteo site's `ALTITUDE`, m) override.

    **Meteo** (`METEO/Meteo_change_1h.dat`): the South Kensington deck's exact format,
    `%2.1f\\t %i\\t %3.2f\\t %2.1f\\t %i\\t %s` for `U`, `Dir`, `Temp`, `Precip`, `Cld`,
    `Fichier` (`NULL`) -- so `wind_speed` must be a multiple of 0.1 m/s and
    `wind_dir_from_deg` a whole number of degrees (written modulo 360); anything else is
    refused rather than rounded. `U` is the speed measured at the meteo site's height;
    `Dir` the direction the wind blows FROM, as the case's. Temperature K -> Celsius (to
    0.01 C), default `_METEO_DEFAULTS`; precipitation and cloud cover from a SIRANE case's
    own raw columns, else `_METEO_DEFAULTS`. Every row ends CRLF. Meteorology must be the
    same at every street within an hour (SIRANE has one station); `meteo_junction` is not
    written.

    **Emissions**: kg/s -> g/s. One street-emission file (`EMISSIONS/EMIS_LIN/Emis_Rues.dat`)
    when every hour is the same, else one file per distinct hour (`Emis_Rues_<n>.dat`),
    named per hour in `EMISSIONS/Emissions_Lin_Surf.dat` with every modulation 1; the
    surface-emission file is header-only (no surface source) and the one point source sits
    100 km outside the network with an all-zero series (as in the South Kensington deck).

    **Species** (`ESPECES/Especes.dat`): the case's species first, in its order, with
    `Flag = 1`, then the rest of `_SPECIES_PROPERTIES` with `Flag = 0`; `Vdepot` and
    `CoeffLessivage` 1 with option `deposition = 1`, 0 with `deposition = 0` -- the default,
    except that a case read from SIRANE writes its own species' flags back. Chemistry
    (`CHAPMAN = 1`) needs NO2, NO and O3 among the species.

    **Background** (`FOND/Concentration_Fond.dat`): kg/m3 -> micrograms/m3, one value per
    hour for the network (so the same at every street), covering whole days from 00:00 of
    the first hour's day to 23:00 of the last's (SIRANE reads it by date; hours outside the
    period repeat the nearest period hour).

    Also written: empty receptor and source-group files, the three grid files (the
    junctions' bounding box plus `_GRID_MARGIN`; the meteo grid 2 x 2, wider still), and
    percentile and threshold files for the active species (statistics only).

    `options`: `chapman` (0/1, `CHAPMAN`), `plume` (0/1, `B_PANACHE`, SIRANE's street-plume
    model above the roofs), `deposition` (0/1), `latitude`, `measurement_height`,
    `input_dir`, `result_dir`, and any `_CARRIED_OVER` keyword. Anything else raises
    `ValueError` naming it.
    """
    out_dir = Path(out_dir)
    native = dict(native or {})
    species = list(species)
    options = check_sirane_options(options, species)
    n_hours, n_streets, n_species = len(times), len(network.streets), len(species)
    for label, value in (("emissions", emissions), ("background", background)):
        if np.shape(value) != (n_hours, n_streets, n_species):
            raise ValueError(
                f"write_case: {label} must have shape {(n_hours, n_streets, n_species)}, "
                f"got {np.shape(value)}"
            )
    for key, value in meteo.items():
        if np.shape(value) != (n_hours, n_streets):
            raise ValueError(
                f"write_case: meteo[{key!r}] must have shape {(n_hours, n_streets)}, got "
                f"{np.shape(value)}"
            )
    if n_hours == 0 or any(abs(t - 3600.0 * k) > 1e-9 for k, t in enumerate(times)):
        raise ValueError(
            f"write_case: SIRANE decks are hourly; times must be 0, 3600, 7200, ... "
            f"(got {list(times)[:5]}...)"
        )
    hours = [start + timedelta(hours=k) for k in range(n_hours)]

    # ------------------------------------------------------------------ master values
    master = dict(_WRITE_DEFAULTS)
    native_options = native.get("options", {})
    master.update({k: native_options[k] for k in _CARRIED_OVER if k in native_options})
    master.update({k: str(v) for k, v in options.items() if k in _CARRIED_OVER})
    for key in _CARRIED_OVER:
        if key in master:
            _check_range(key, master[key], where="master-file setting")
    if int(float(master["CHAPMAN"])) == 1:
        missing = [sp for sp in _CHAPMAN_SPECIES if sp not in species]
        if missing:
            raise ValueError(
                f"write_case: chemistry (CHAPMAN = 1) acts on NO2, NO and O3; the case's "
                f"species {species} lack {missing}"
            )

    input_dir = str(options.get("input_dir", out_dir.name)).replace("\\", "/").strip("/")
    result_dir = str(options.get("result_dir", f"{input_dir}/{RESULT_SUBDIR}")).replace(
        "\\", "/").strip("/")
    input_parts = [p for p in input_dir.split("/") if p not in ("", ".")]
    resolved = out_dir.resolve()
    if (not input_parts or ".." in input_parts
            or [p.casefold() for p in resolved.parts[-len(input_parts):]]
            != [p.casefold() for p in input_parts]):
        raise ValueError(
            f"write_case: input_dir {input_dir!r} must be the trailing part of out_dir "
            f"({out_dir}): SIRANE finds the deck at <its working directory>/{input_dir}"
        )
    working_dir = resolved.parents[len(input_parts) - 1]
    if Path(result_dir).is_absolute() or ".." in result_dir.split("/"):
        raise ValueError(
            f"write_case: result_dir {result_dir!r} must be a relative path below SIRANE's "
            f"working directory"
        )

    z0_values = {s.z0_b for s in network.streets}
    if len(z0_values) != 1:
        raise ValueError(
            f"write_case: the streets' z0_b take {len(z0_values)} values; SIRANE has one "
            f"building roughness (Z0D_BAT) for the network"
        )
    z0_b = z0_values.pop()
    _check_range("Z0D_BAT", z0_b, where="the streets' z0_b")

    # ------------------------------------------------------------------ network
    ids = [str(j) for j in junction_ids]
    if not (len(ids) == len(network.junctions) and all(j.isdigit() for j in ids)
            and len(set(ids)) == len(ids)):
        ids = [str(i + 1) for i in range(len(network.junctions))]
    node_id = dict(zip(network.junctions, ids, strict=True))
    native_streets = native.get("streets") or []
    lines, records = [], []
    for i, street in enumerate(network.streets):
        p, q = ((network.x[street.u], network.y[street.u]),
                (network.x[street.v], network.y[street.v]))
        distance = math.hypot(q[0] - p[0], q[1] - p[1])
        if abs(distance - street.length) > 1e-6 * max(1.0, street.length):
            raise ValueError(
                f"write_case: street {street.name!r} has length {street.length} m but its "
                f"junctions are {distance} m apart; SIRANE takes a street's length from its "
                f"geometry"
            )
        record = {"TYPE": 0, "NDDEB": node_id[street.u], "NDFIN": node_id[street.v],
                  "WG": street.width / 2.0, "WD": street.width / 2.0,
                  "HG": street.height, "HD": street.height, "MODUL_EMIS": 0}
        if len(native_streets) == n_streets:
            own = native_streets[i]
            wg, wd, hg, hd = (float(own[f]) for f in ("WG", "WD", "HG", "HD"))
            if (abs(wg + wd - street.width) <= 1e-9 * street.width
                    and abs(0.5 * (hg + hd) - street.height) <= 1e-9 * street.height):
                record.update(TYPE=int(float(own["TYPE"])), WG=wg, WD=wd, HG=hg, HD=hd)
        lines.append([p, q])
        records.append(record)

    # ------------------------------------------------------------------ meteo
    for key in ("wind_dir_from_deg", "wind_speed"):
        if key not in meteo:
            raise ValueError(f"write_case: format='sirane' needs meteo[{key!r}]")
    direction = _uniform("meteo['wind_dir_from_deg']",
                         np.asarray(meteo["wind_dir_from_deg"], dtype=np.float64),
                         why="one wind direction")
    speed = _uniform("meteo['wind_speed']", np.asarray(meteo["wind_speed"], dtype=np.float64),
                     why="one wind speed")
    if "temperature" in meteo:
        temperature_c = _uniform("meteo['temperature']",
                                 np.asarray(meteo["temperature"], dtype=np.float64),
                                 why="one temperature") - CELSIUS_TO_K
    else:
        temperature_c = np.full(n_hours, _METEO_DEFAULTS["temperature_c"], dtype=np.float64)
    raw = native.get("meteo_raw", {})
    precip = (list(raw["precip"]) if len(raw.get("precip", [])) == n_hours
              else [_METEO_DEFAULTS["precip"]] * n_hours)
    cloud = (list(raw["cloud"]) if len(raw.get("cloud", [])) == n_hours
             else [_METEO_DEFAULTS["cloud"]] * n_hours)
    meteo_lines = ["Date\tU\tDir\tTemp\tPrecip\tCld\tFichier"]
    for k, when in enumerate(hours):
        if not _whole(float(direction[k]), step=1.0):
            raise ValueError(
                f"write_case: wind direction {direction[k]} deg at {when.isoformat()} is not "
                f"a whole number of degrees; SIRANE's meteo file (the study's format) writes "
                f"it as an integer -- round it first"
            )
        if not _whole(float(speed[k]), step=0.1):
            raise ValueError(
                f"write_case: wind speed {speed[k]} m/s at {when.isoformat()} is not a "
                f"multiple of 0.1 m/s; SIRANE's meteo file (the study's format) writes one "
                f"decimal -- round it first"
            )
        dir_int = int(round(float(direction[k]))) % 360
        meteo_lines.append(
            f"{_date_table(when)}\t{float(speed[k]):2.1f}\t {dir_int:d}\t "
            f"{float(temperature_c[k]):3.2f}\t {float(precip[k]):2.1f}\t "
            f"{int(round(float(cloud[k]))):d}\t NULL"
        )

    # ------------------------------------------------------------------ species
    native_species = native.get("species_table", {})
    unknown_species = [sp for sp in species if sp not in _SPECIES_PROPERTIES]
    if unknown_species:
        raise ValueError(
            f"write_case: species {unknown_species} are not SIRANE species this writer "
            f"knows ({sorted(_SPECIES_PROPERTIES)}); write a passive tracer as one of them "
            f"with chemistry and deposition off"
        )
    deposition = options.get("deposition")
    species_lines = ["Id\tFlag\tMmolaire\tVdepot\tCoeffLessivage\tDiamPart\tRhoPart"]
    for name in species + [sp for sp in _SPECIES_PROPERTIES if sp not in species]:
        molar, diameter, density = _SPECIES_PROPERTIES[name]
        if deposition is not None:
            vdepot = lessivage = str(deposition)
        elif name in native_species:
            vdepot = str(native_species[name].get("Vdepot", "0"))
            lessivage = str(native_species[name].get("CoeffLessivage", "0"))
        else:
            vdepot = lessivage = "0"
        flag = "1" if name in species else "0"
        species_lines.append("\t".join(
            [name, flag, _num(molar), vdepot, lessivage, _num(diameter), _num(density)]))

    # ------------------------------------------------------------------ background
    background_ug = _uniform("background", np.asarray(background, dtype=np.float64),
                             why="one background concentration") * UG_PER_KG
    first_day = datetime(hours[0].year, hours[0].month, hours[0].day)
    last_day = datetime(hours[-1].year, hours[-1].month, hours[-1].day)
    n_cover = int((last_day - first_day).total_seconds() // 3600) + 24
    cover = [first_day + timedelta(hours=k) for k in range(n_cover)]
    background_lines = ["Date\t" + "\t".join(species)]
    for when in cover:
        k = min(max(int((when - hours[0]).total_seconds() // 3600), 0), n_hours - 1)
        background_lines.append(_date_table(when) + "\t"
                                + "\t".join(_num(v) for v in background_ug[k]))

    # ------------------------------------------------------------------ emissions
    emissions_g = np.asarray(emissions, dtype=np.float64) * G_PER_KG
    distinct: list[np.ndarray] = []
    hour_file: list[int] = []
    for k in range(n_hours):
        for n, pattern in enumerate(distinct):
            if np.array_equal(pattern, emissions_g[k]):
                hour_file.append(n)
                break
        else:
            distinct.append(emissions_g[k])
            hour_file.append(len(distinct) - 1)
    street_files = ([f"{_STREET_EMISSION_DIR}/Emis_Rues.dat"] if len(distinct) == 1 else
                    [f"{_STREET_EMISSION_DIR}/Emis_Rues_{n}.dat" for n in range(len(distinct))])
    mod_lin = "\t".join(f"Mod_Lin_0_{sp}" for sp in species)
    mod_surf = "\t".join(f"Mod_Surf_{sp}" for sp in species)
    ones = "\t".join("1" for _ in species)
    evolution_lines = [f"Date\tFich_Lin\t{mod_lin}\tFich_Surf\t{mod_surf}"]
    for k, when in enumerate(hours):
        evolution_lines.append(
            f"{_date_unpadded(when)}\t{street_files[hour_file[k]]}\t{ones}\t"
            f"{_SURFACE_EMISSION_FILE}\t{ones}")

    # ------------------------------------------------------------------ extent
    xs = [network.x[j] for j in network.junctions]
    ys = [network.y[j] for j in network.junctions]
    x0, x1 = math.floor(min(xs)) - _GRID_MARGIN, math.ceil(max(xs)) + _GRID_MARGIN
    y0, y1 = math.floor(min(ys)) - _GRID_MARGIN, math.ceil(max(ys)) + _GRID_MARGIN
    mx0, mx1 = math.floor((x0 - 50) / 100) * 100, math.ceil((x1 + 50) / 100) * 100
    my0, my1 = math.floor((y0 - 50) / 100) * 100, math.ceil((y1 + 50) / 100) * 100

    # ------------------------------------------------------------------ sites
    sites = {}
    for which in ("site_meteo", "site_disp"):
        values = dict(_SITE_DEFAULTS, ZDISPL=_DISPLACEMENT[which])
        values.update(native.get(which, {}))
        if "latitude" in options:
            values["LATITUDE"] = float(options["latitude"])
        sites[which] = values
    if "measurement_height" in options:
        sites["site_meteo"]["ALTITUDE"] = float(options["measurement_height"])
    for which, values in sites.items():
        for key, value in values.items():
            _check_range(key, value, where=f"{which} value")

    # ------------------------------------------------------------------ write
    files = _DECK_FILES
    shown_in_calcul = [k for k in _CARRIED_OVER if k in master and k not in
                       {"CHAPMAN", "B_PANACHE", "B_FOND", "H_R", "U_MIN", "SIGMA_V_MIN",
                        "SIGMA_W_MIN", "AFFICH"}]
    sections = [
        ("Periode", [("DATE_DEB", _date_master(hours[0])),
                     ("DATE_FIN", _date_master(hours[-1]))]),
        ("Polluants", [("FICH_ESPECES", files["FICH_ESPECES"]),
                       ("FICH_SRCE_GROUP", files["FICH_SRCE_GROUP"]),
                       ("CHAPMAN", master["CHAPMAN"]),
                       ("B_EMIS_NOEQNO2", master["B_EMIS_NOEQNO2"])]),
        ("Emissions", [("FICH_SOURCES_PONCT", files["FICH_SOURCES_PONCT"]),
                       ("FICH_EMIS_RUE", files["FICH_EMIS_RUE"]),
                       ("I_N_MOD_LIN", master["I_N_MOD_LIN"]),
                       ("FICH_POLL_FOND", files["FICH_POLL_FOND"])]),
        ("Caracteristiques du milieu urbain",
         [("B_RUE_DECOUP", master["B_RUE_DECOUP"]),
          ("TYPE_FICH_RESEAU", master["TYPE_FICH_RESEAU"]),
          ("FICH_RESEAU", files["FICH_RESEAU"]),
          ("FICH_SITE_DISP", files["FICH_SITE_DISP"]),
          ("Z0D_BAT", _num(z0_b)), ("H_R", master["H_R"])]),
        ("Conditions meteorologiques",
         [("TYPE_METEO", master["TYPE_METEO"]), ("FICH_METEO", files["FICH_METEO"]),
          ("FICH_SITE_METEO", files["FICH_SITE_METEO"]),
          ("B_STREET_U_SIGMA_W", master["B_STREET_U_SIGMA_W"]),
          ("U_MIN", master["U_MIN"]), ("SIGMA_V_MIN", master["SIGMA_V_MIN"]),
          ("SIGMA_W_MIN", master["SIGMA_W_MIN"])]),
        ("Grilles", [("FICH_GRD_MET", files["FICH_GRD_MET"]),
                     ("FICH_GRD_SORTIE", files["FICH_GRD_SORTIE"]),
                     ("FICH_GRD_EMIS_SURF", files["FICH_GRD_EMIS_SURF"])]),
        ("Sortie des resultats",
         [(k, master[k]) for k in ("AFFICH", "B_CALC_STAT")]
         + [("FICH_RECEPT", files["FICH_RECEPT"])]
         + [(k, master[k]) for k in ("B_CONC_JOUR", "CALC_GRID", "B_ECRIRE_DEPOT",
                                     "CALC_RUES", "FORMAT_RUES_SORTIE", "B_CALC_INTERSECT",
                                     "FORMAT_CHP_SORTIE", "FORMAT_IMAGE_SORTIE")]
         + [("FICH_DIR_RESUL", result_dir)]),
        ("Parametres statistiques", [("FICH_PERCENT", files["FICH_PERCENT"]),
                                     ("FICH_SEUILS", files["FICH_SEUILS"])]),
        ("Parametres de calcul",
         [(k, master[k]) for k in ("B_PANACHE", "B_FOND")
          if int(float(master[k])) != 1]
         + [(k, master[k]) for k in shown_in_calcul]),
    ]
    master_lines = ["/" + "*" * 42, "/** Donnees pour l'utilisation de SIRANE **",
                    "/** ecrites par noodl physics            **", "/" + "*" * 42,
                    f"{_MASTER_LABELS['FICH_DIR_INPUT'][0]} = {input_dir}"]
    for title, values in sections:
        master_lines += _labelled(title, values, _MASTER_LABELS)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / MASTER_FILE
    _write_lines(master_path, master_lines)
    for sub in ("", *RESULT_SUBFOLDERS):
        (working_dir / result_dir / sub).mkdir(parents=True, exist_ok=True)

    for which, key, title in (("site_disp", "FICH_SITE_DISP", "Caracteristiques du quartier"),
                              ("site_meteo", "FICH_SITE_METEO",
                               "Caracteristiques du site meteo")):
        site_lines = [f"/ {title} :", "/" + "-" * 31]
        site_lines += [f"{_SITE_LABELS[k][0]} = {float(sites[which][k])!r}"
                       for k in _SITE_LABELS if k in sites[which]]
        _write_lines(out_dir / files[key], site_lines)

    _shapefile.write_polylines(out_dir / files["FICH_RESEAU"], lines, records, _DBF_FIELDS)
    _write_lines(out_dir / files["FICH_METEO"], meteo_lines)
    _write_lines(out_dir / files["FICH_ESPECES"], species_lines)
    _write_lines(out_dir / files["FICH_SRCE_GROUP"], ["Id"])
    _write_lines(out_dir / files["FICH_POLL_FOND"], background_lines)
    _write_lines(out_dir / files["FICH_EMIS_RUE"], evolution_lines)
    for name, pattern in zip(street_files, distinct, strict=True):
        _write_lines(out_dir / name, ["Id\t" + "\t".join(species)]
                     + [f"{i}\t" + "\t".join(_num(v) for v in pattern[i])
                        for i in range(n_streets)])
    _write_lines(out_dir / _SURFACE_EMISSION_FILE, ["X\tY\t" + "\t".join(species)],
                 final_newline=False)
    _write_lines(out_dir / files["FICH_SOURCES_PONCT"],
                 ["Id\tX\tY\tZ\tD\tUx\tUy\tUz\tT\tFichier",
                  f"id\t{_num(mx0 - 100_000)}\t{_num(my0 - 100_000)}\t0\t0\t0\t0\t0\t15\t"
                  f"{_POINT_SERIES_FILE}"])
    _write_lines(out_dir / _POINT_SERIES_FILE,
                 ["Date\t" + "\t".join(species)]
                 + [f"{_date_unpadded(when)}\t" + "\t".join("0" for _ in species)
                    for when in cover])
    _write_lines(out_dir / files["FICH_RECEPT"], ["Id\tX\tY\tZ\tType\tFichier"])
    grid_header = "Nx\tNy\txmin\txmax\tymin\tymax"
    for key, row in (("FICH_GRD_MET", (2, 2, mx0, mx1, my0, my1)),
                     ("FICH_GRD_SORTIE", (100, 100, x0, x1, y0, y1)),
                     ("FICH_GRD_EMIS_SURF", (1, 1, x0, x1, y0, y1))):
        _write_lines(out_dir / files[key], [grid_header, "\t".join(_num(v) for v in row)],
                     final_newline=False)
    stats_header = "\t".join(species)
    _write_lines(out_dir / files["FICH_PERCENT"],
                 [stats_header] + ["\t".join(p for _ in species) for p in ("50", "98")])
    _write_lines(out_dir / files["FICH_SEUILS"],
                 [stats_header] + ["\t".join(t for _ in species) for t in ("40", "200")])
    return master_path


# ------------------------------------------------------------------------------- sweeps

def write_sweep_manifest(out_dir: Path, *, runs: list[dict]) -> None:
    """Writes a sweep's `runs.csv` into `out_dir`: one row per deck `write_sweep` wrote under
    `out_dir/decks/<run-id>/` (`runs`: one dict per deck, with `run_id`, `direction_deg`,
    `speed`, `source_street`, `source_index`, `variant`, `options`), giving its id, wind
    direction (degrees FROM, clockwise from north), speed (m/s), source street (SIRANE's
    street number = the network's record order, and the case's own street id), variant and
    the options its deck was written with."""
    out_dir = Path(out_dir)
    csv = ["run_id,direction_deg,speed_m_s,source_index,source_street,variant,options"]
    for run in runs:
        options = ";".join(f"{k}={v}" for k, v in run["options"].items())
        csv.append(f"{run['run_id']},{_num(run['direction_deg'])},{_num(run['speed'])},"
                   f"{run['source_index']},{run['source_street']},{run['variant']},{options}")
    _write_lines(out_dir / "runs.csv", csv)
