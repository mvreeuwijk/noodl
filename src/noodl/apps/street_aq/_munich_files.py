"""MUNICH's own street-network file format: `munich.cfg` plus the street/intersection
tables, the `[section]`/`key: value` config dialect, and the `float32` binaries it names.

Private to `noodl.apps.street_aq`: `case.py` is the format-neutral public surface
(`StreetCase`, `read_case`, `write_case`, `drivers_at`); this module knows nothing about
that dataclass -- it reads and writes plain dicts/arrays in the neutral vocabulary
(`wind_dir_from_deg`, `wind_speed`, `h_abl`, `u_star`, `lmo`, `temperature`; kg/s; kg/m3),
and every MUNICH name, unit and convention is translated here, in both directions.

The schema here is the one discovered by reading the MUNICH v2.2 source
(github.com/cerea-lab/munich) and its shipped `processing/photochemistry` example, NOT the
full Talos config grammar -- this module reads and writes exactly what `write_munich_case`
produces (one `key: value` or `key = value` pair per line, or a bare `key value` override
line, inside `[section]` blocks), which is also what MUNICH's own `Talos::ConfigStream`
accepts (verified by running MUNICH v2.2's own `munich` binary on this writer's output).

MUNICH's own files are in micrograms (emissions in micrograms/s per street, concentrations
in micrograms/m3 -- matching `StreetNetworkTransport.cxx:2536`'s
`emission_rate = street->GetEmission(s); // ug/s`), but `read_munich_case`'s result is
already SI (kg/s, kg/m3): converted at read time, dividing by `UG_PER_KG` (and
`write_munich_case` multiplies back up at write time), so nothing outside this module needs
to know MUNICH's native units.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from noodl.apps.street_aq.network import Street, StreetNetwork

EARTH_RADIUS_M = 6371229.0
"""`StreetNetworkTransport.cxx:18`'s `earth_radius` -- the constant MUNICH itself uses to
turn a street's endpoint lon/lat into metres. Reusing it here means a network written by
`write_munich_case` and re-projected internally by MUNICH recovers the same `x`, `y` metres
this reader would compute directly."""

UG_PER_KG = 1e9
"""MUNICH's native mass unit (micrograms) per kg (noodl's SI unit). `read_munich_case`
divides emissions/background by this AT READ TIME; `write_munich_case` multiplies by it at
write time."""

_METEO_FIELDS = {
    "WindSpeed": "wind_speed",
    "PBLH": "h_abl",
    "UST": "u_star",
    "LMO": "lmo",
    "SurfaceTemperature": "temperature",
}
"""MUNICH meteo field name -> the neutral meteo key it becomes. `WindDirection` is handled
separately because it needs the degrees-FROM conversion (`munich_rad_to_deg_from`)."""

_INTER_SUFFIX = "Inter"

_STREET_COLUMNS = ("id", "begin_inter", "end_inter", "length", "width", "height", "typo")
"""`street.dat`'s columns, in order -- the layout MUNICH's shipped cases use."""

DEFAULT_LAT0_DEG = 48.85
DEFAULT_LON0_DEG = 2.35
"""Where `write_munich_case` anchors a network whose metres are LOCAL (a synthetic case):
its `(x, y) = (0, 0)` goes to this lon/lat. Only the network's own extent matters to
MUNICH's geometry, so any mid-latitude anchor works; a case read from MUNICH files carries
its own projection instead (see `read_munich_case`'s `projection`)."""


def munich_rad_to_deg_from(direction_rad) -> np.ndarray:
    """MUNICH's wind direction (radians) -> degrees clockwise from north, FROM.

    MUNICH's `WindDirection` is radians clockwise from north, and it is the direction the
    wind blows TOWARD, not the meteorological FROM. MUNICH's own preprocessor says so where
    it builds the field from WRF's (u, v), `preprocessing/meteo.py:380-386` and
    `compute_wdir`, `:814-818`: `atan2(v, u)` turned into a clockwise-from-north bearing,
    "0 for the wind to north, pi/2 to east, pi to south". The transport code agrees:
    `ComputeIntersectionFlux` (`StreetNetworkTransport.cxx:2872-2882`) calls a street whose
    bearing away from the junction lies within 90 degrees of `WindDirection` an OUTFLOW,
    which is right only for a TOWARD direction. Degrees-FROM is therefore
    `WindDirection + 180` degrees.
    """
    return (np.degrees(np.asarray(direction_rad, dtype=np.float64)) + 180.0) % 360.0


def deg_from_to_munich_rad(deg_from) -> np.ndarray:
    """The inverse of `munich_rad_to_deg_from` -- what a MUNICH `WindDirection`/
    `WindDirectionInter` binary field holds (radians TOWARD) for a degrees-FROM direction."""
    return np.radians((np.asarray(deg_from, dtype=np.float64) + 180.0) % 360.0)


def parse_date(text: str, *, where: str) -> datetime:
    """A MUNICH/Talos date string -> `datetime`. Talos reads the DIGITS of the string, in
    the order year (4), month, day, hour, minute, second (2 each), ignoring separators --
    `2014-03-16-00`, `2014-03-16_00-00-00` and `2014031600` are the same hour. `where`
    names the file/section for the error message."""
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) not in (8, 10, 12, 14):
        raise ValueError(
            f"read_case: {where} Date_min {text!r} is not a date "
            f"(expected YYYY-MM-DD[-HH[-MM[-SS]]], any separators)"
        )
    parts = [int(digits[:4])] + [int(digits[i:i + 2]) for i in range(4, len(digits), 2)]
    return datetime(*parts)


def format_date(start: datetime) -> str:
    """`start` as a MUNICH `Date_min`: `YYYY-MM-DD-HH` on the hour (the form verified against
    the MUNICH v2.2 binary), `YYYY-MM-DD_HH-MM-SS` otherwise."""
    if start.minute == 0 and start.second == 0 and start.microsecond == 0:
        return start.strftime("%Y-%m-%d-%H")
    return start.strftime("%Y-%m-%d_%H-%M-%S")


def _parse_cfg(path: Path) -> dict[str, dict[str, str]]:
    """A minimal `[section]` / `key: value` (or `key = value`, or a bare `key value`
    override line) parser -- see the module docstring for what dialect this covers."""
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections[current] = {}
            continue
        if current is None:
            continue
        if ":" in line:
            key, _, value = line.partition(":")
        elif "=" in line:
            key, _, value = line.partition("=")
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        sections[current][key.strip()] = value.strip()
    return sections


def _is_num(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _read_binary_field(path: Path, nt: int, n_columns: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4").astype(np.float64)
    expected = nt * n_columns
    if data.size != expected:
        raise ValueError(
            f"read_case: {path} has {data.size} float32 values, expected "
            f"{nt} x {n_columns} = {expected}"
        )
    return data.reshape(nt, n_columns)


def _section_value(section: Mapping[str, str], key: str, *, where: str) -> str:
    """`section[key]`, or a `ValueError` naming the section and the missing key."""
    try:
        return section[key]
    except KeyError:
        raise ValueError(f"read_case: {where} has no {key!r} entry") from None


def _resolve_field(
    section: Mapping[str, str],
    field: str,
    *,
    where: str,
    root: Path,
    n_columns: int,
    nt: int,
    start: datetime,
    delta_t: float,
) -> np.ndarray:
    """The domain's `(nt, n_columns)` rows of `section`'s `field`, starting at the domain's
    own `start` and stepping by its `delta_t`.

    The field's value is an override line, or the section's generic `Filename` template with
    `&f` replaced. A value that parses as a number is a constant broadcast to every row
    (MUNICH's own `InputFiles::Read`/`is_num` shortcut). Otherwise it names a float32 binary
    of `(section Nt, n_columns)` records, relative to `root`, that begins at the SECTION's
    own `Date_min` and steps by the section's own `Delta_t` -- MUNICH locates the domain's
    rows by date, so the domain's first row is record
    `(domain Date_min - section Date_min) / Delta_t`. A one-record binary (`Nt: 1`) is
    broadcast to every row, as a constant is. Anything else must cover the domain exactly
    on its own time grid: a different `Delta_t`, a start that is not a whole number of steps
    at or before the domain's, or too few records raises `ValueError` naming the section,
    the field and both sizes.
    """
    raw = section.get(field)
    if raw is None:
        raw = _section_value(section, "Filename", where=where).replace("&f", field)
    if _is_num(raw):
        return np.full((nt, n_columns), float(raw), dtype=np.float64)
    n_records = int(float(_section_value(section, "Nt", where=where)))
    data = _read_binary_field(root / raw, n_records, n_columns)
    if n_records == 1:
        return np.repeat(data, nt, axis=0)
    section_dt = float(_section_value(section, "Delta_t", where=where))
    if section_dt != delta_t:
        raise ValueError(
            f"read_case: {where} field {field!r} has Delta_t {section_dt!r} s, but the "
            f"domain's is {delta_t!r} s; this reader does not resample"
        )
    section_start = parse_date(_section_value(section, "Date_min", where=where), where=where)
    lag = (start - section_start).total_seconds() / delta_t
    offset = int(round(lag))
    if lag < 0 or abs(lag - offset) > 1e-9:
        raise ValueError(
            f"read_case: {where} starts at {section_start.isoformat()}, which is not a whole "
            f"number of {delta_t!r} s steps at or before the domain's start "
            f"{start.isoformat()}"
        )
    if offset + nt > n_records:
        raise ValueError(
            f"read_case: {where} field {field!r} has {n_records} records from "
            f"{section_start.isoformat()}; the domain needs {nt} from record {offset} "
            f"(i.e. {offset + nt}), and only a 1-record field or a constant is broadcast"
        )
    return data[offset:offset + nt]


def _read_species_list(path: Path) -> list[str]:
    section = None
    species: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if section == "species":
            species.extend(line.split())
    return species


def _read_semicolon_table(path: Path) -> list[list[str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f for f in line.split(";") if f != ""]
        rows.append(fields)
    return rows


def _write_binary(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype=np.float64).astype("<f4").tofile(path)


# ------------------------------------------------------------------------------- reading

def read_munich_case(root: Path) -> dict:
    """Every array here is already SI (converting MUNICH's own micrograms at read time --
    see `UG_PER_KG`), and every per-street array is kept PER STREET -- reduction to whatever
    a particular `Model` needs is `noodl.apps.street_aq.case.drivers_at`'s job, not the
    reader's, so the same `StreetCase` can drive a uniform or a per-street model.

    Returns a plain dict of the fields `StreetCase` takes: `network`, `times`, `start`,
    `street_ids`, `junction_ids` (in `network.junctions` order), `species`, `meteo`,
    `meteo_junction`, `emissions`, `background`, `native` (one dict per `munich.cfg`
    section, plus `"projection"`: the `lat_ref_deg`/`lat0_deg`/`lon0_deg` this reader
    projected lon/lat to metres with, so that `write_munich_case` can invert it exactly).
    """
    root = Path(root)
    cfg = _parse_cfg(root / "munich.cfg")
    domain = cfg["domain"]
    nt = int(float(domain["Nt"]))
    delta_t = float(domain["Delta_t"])
    start = parse_date(domain["Date_min"], where="munich.cfg [domain]")
    times = [i * delta_t for i in range(nt)]

    species = _read_species_list(root / domain["Species"])

    street_section = cfg["street"]
    street_path = root / street_section["Street"]
    intersection_path = root / street_section["Intersection"]
    street_rows = _read_semicolon_table(street_path)
    intersection_rows = _read_semicolon_table(intersection_path)
    for row in street_rows:
        if len(row) != len(_STREET_COLUMNS):
            raise ValueError(
                f"read_case: {street_path.name} row {';'.join(row)!r} has {len(row)} "
                f"columns; this reader expects the {len(_STREET_COLUMNS)} columns "
                f"{';'.join(_STREET_COLUMNS)}"
            )
    for row in intersection_rows:
        if len(row) < 3:
            raise ValueError(
                f"read_case: {intersection_path.name} row {';'.join(row)!r} has "
                f"{len(row)} columns; expected at least id;lon;lat"
            )

    coordinates = {row[0]: (float(row[1]), float(row[2])) for row in intersection_rows}
    col_of_intersection_id = {row[0]: i for i, row in enumerate(intersection_rows)}
    used_ids = {row[1] for row in street_rows} | {row[2] for row in street_rows}
    lats = [coordinates[i][1] for i in used_ids if i in coordinates]
    lat0 = sum(lats) / len(lats) if lats else 0.0
    lat0_rad = math.radians(lat0)

    def project(intersection_id: str) -> tuple[float, float]:
        lon, lat = coordinates[intersection_id]
        x = EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(lon)
        y = EARTH_RADIUS_M * math.radians(lat)
        return x, y

    x: dict[str, float] = {}
    y: dict[str, float] = {}
    raw_junction_id: dict[str, str] = {}
    streets = []
    street_ids: list[str] = []
    for street_id, begin_id, end_id, length, width, height, _typo in street_rows:
        u, v = f"j{begin_id}", f"j{end_id}"
        for jid, name in ((begin_id, u), (end_id, v)):
            if name not in x:
                x[name], y[name] = project(jid)
                raw_junction_id[name] = jid
        streets.append(Street(street_id, u, v, float(length), float(width), float(height)))
        street_ids.append(street_id)
    network = StreetNetwork(streets=streets, x=x, y=y)
    n_streets = len(street_ids)
    junctions = network.junctions
    junction_ids = [raw_junction_id[name] for name in junctions]
    junction_cols = [col_of_intersection_id[jid] for jid in junction_ids]
    n_intersections = len(intersection_rows)

    data_name = cfg["data"]["Data_description"]
    data_cfg = _parse_cfg(root / data_name)

    def field(section: str, name: str, n_columns: int) -> np.ndarray:
        return _resolve_field(
            data_cfg[section], name, where=f"{data_name} [{section}]", root=root,
            n_columns=n_columns, nt=nt, start=start, delta_t=delta_t,
        )

    emissions = np.zeros((nt, n_streets, len(species)), dtype=np.float64)
    background = np.zeros((nt, n_streets, len(species)), dtype=np.float64)
    for s, sp in enumerate(species):
        emissions[:, :, s] = field("emission", sp, n_streets)
        # Per-street, exactly like emission and meteo (`StreetNetworkTransport.cxx:600`,
        # `Background_i.Resize(GridS2D, GridST2D)`).
        background[:, :, s] = field("background_concentration", sp, n_streets)
    emissions /= UG_PER_KG   # MUNICH's micrograms/s -> kg/s.
    background /= UG_PER_KG  # MUNICH's micrograms/m3 -> kg/m3.

    fields_available = set(data_cfg["meteo"].get("Fields", "").split())

    def street_field(name: str) -> np.ndarray:
        return field("meteo", name, n_streets)

    def junction_field(name: str) -> np.ndarray:
        return field("meteo", name, n_intersections)[:, junction_cols]

    meteo: dict[str, np.ndarray] = {}
    if "WindDirection" in fields_available:
        meteo["wind_dir_from_deg"] = munich_rad_to_deg_from(street_field("WindDirection"))
    for munich_name, key in _METEO_FIELDS.items():
        if munich_name in fields_available:
            meteo[key] = street_field(munich_name)

    meteo_junction: dict[str, np.ndarray] = {}
    if f"WindDirection{_INTER_SUFFIX}" in fields_available:
        meteo_junction["wind_dir_from_deg"] = munich_rad_to_deg_from(
            junction_field(f"WindDirection{_INTER_SUFFIX}")
        )
    for munich_name, key in _METEO_FIELDS.items():
        inter_name = f"{munich_name}{_INTER_SUFFIX}"
        if inter_name in fields_available:
            meteo_junction[key] = junction_field(inter_name)

    native: dict = {name: dict(values) for name, values in cfg.items()}
    native["projection"] = {"lat_ref_deg": lat0, "lat0_deg": 0.0, "lon0_deg": 0.0}

    return dict(
        network=network, times=times, start=start, street_ids=street_ids,
        junction_ids=junction_ids, species=species, meteo=meteo,
        meteo_junction=meteo_junction, emissions=emissions, background=background,
        native=native,
    )


# ------------------------------------------------------------------------------- results

def read_munich_results(results_dir: Path, *, case) -> dict:
    """Reads MUNICH's own `results/<species>.bin` outputs into the fields `StreetResults`
    takes: `times`, `street_ids`, `species`, `c_in` (`c_above`, `u_canyon`, `sigma_w_roof`,
    `u_exchange` and `meteo` are the ones only SIRANE fills -- `{}`/`None` here).

    `write_munich_case`'s own saver config (`munich-saver.cfg`) templates
    `Output_file: <Result_dir>/&f.bin` with `<Result_dir>` = `results/` and `&f` the species
    name, so each active species writes its own `results/<species>.bin`: `float32`,
    row-major `(n_hours, n_streets)`, micrograms/m3 -- MUNICH's own concentration unit, like
    its emission/background inputs (see the module docstring) -- converted to kg/m3 here.

    `case` supplies everything a MUNICH binary itself does not carry: the street order and
    count (`case.street_ids`), species (`case.species`), hour count (`len(case.times)`) and
    absolute times (`case.start` + `case.times` -- `case.start` must not be `None`). A
    species file that does not exist, or whose size is not `n_hours * n_streets`, raises
    naming the path and the sizes.
    """
    results_dir = Path(results_dir)
    if case.start is None:
        raise ValueError(
            "read_results: case.start is None; a MUNICH result needs the case's own "
            "absolute times (case.start + case.times)"
        )
    n_hours, n_streets = len(case.times), len(case.street_ids)
    c_in: dict[str, np.ndarray] = {}
    for sp in case.species:
        path = results_dir / f"{sp}.bin"
        if not path.is_file():
            raise FileNotFoundError(
                f"read_results: {path} does not exist; every case species needs a MUNICH "
                f"results/<species>.bin"
            )
        data = np.fromfile(path, dtype="<f4").astype(np.float64)
        expected = n_hours * n_streets
        if data.size != expected:
            raise ValueError(
                f"read_results: {path} has {data.size} float32 value(s), expected "
                f"n_hours x n_streets = {n_hours} x {n_streets} = {expected}"
            )
        c_in[sp] = data.reshape(n_hours, n_streets) / UG_PER_KG

    times = [case.start + timedelta(seconds=t) for t in case.times]
    return dict(
        times=times, street_ids=list(case.street_ids), species=list(case.species),
        c_in=c_in, c_above={}, u_canyon=None, sigma_w_roof=None, u_exchange=None, meteo={},
    )


# ------------------------------------------------------------------------------- writing

_DEFAULT_STREET_OPTIONS: dict[str, str] = {
    "Mean_wind_speed_parameterization": "Exponential",
    "Transfer_parameterization": "Schulte",
    "Building_height_wind_speed_parameterization": "Sirane",
    "Compute_Macdonald_from": "Ustar",
    "Deposition_wind_profile": "Masson",
    "With_horizontal_fluctuation": "yes",
    "With_stationary_hypothesis": "yes",
    "Numerical_method_parameterization": "ETR",
    "Intersection": "intersection.dat",
    "Street": "street.dat",
    "Minimum_Street_Wind_Speed": "0.1",
    "With_local_data": "yes",
    "Sub_delta_t_min": "1.0",
    "Building_density": "0.4",
    "With_tree_aerodynamic": "no",
    "With_tree_deposition": "no",
    "Zref": "30.0",
}
"""The `[street]` closure options `write_munich_case` writes by default -- MUNICH's
equivalents of `canyon_wind="exponential"`, `exchange="schulte"`, `roof_wind_form="sirane"`,
`direction_averaging="munich"` (`With_horizontal_fluctuation`) and
`With_stationary_hypothesis: yes` (each hour's steady street balance by fixed-point
iteration, rather than MUNICH's default explicit integrator, which is unstable at hour-long
steps on short streets) -- plus the keys MUNICH v2.2 requires regardless of the options
actually exercised (`Compute_Macdonald_from`, `Deposition_wind_profile`, `Sub_delta_t_min`,
`Building_density`, `With_tree_aerodynamic`, `With_tree_deposition`; the shipped example
predates them). A case read from MUNICH files writes its own `[street]` section back over
these; `write_munich_case`'s `options` override both."""

_FILE_KEYS = ("Street", "Intersection")
"""`[street]` keys naming files this writer itself writes -- never taken from a read case."""

_REQUIRED_METEO_DEFAULTS: dict[str, float] = {
    "Rain": 0.0,
    "SolarRadiation": 0.0,
    "SpecificHumidity": 0.01,
    "SurfacePressure": 101325.0,
    "SurfaceTemperature": 293.15,
    "Attenuation": 1.0,
}
"""`[meteo]` fields MUNICH v2.2 requires listed in `Fields` even with chemistry, deposition
and scavenging off -- `StreetNetworkTransport.cxx:750-776`'s unconditional `InitData` calls
for `SurfaceTemperature`, `SurfacePressure`, `Rain`, `SpecificHumidity`, and
`StreetNetworkChemistry.cxx:744-751`'s for `Attenuation` (read whenever `With_local_data` is
`yes`, this writer's own default); without them MUNICH stops with `undefined variable
SurfacePressure`. `write_munich_case` always writes all six; a value the case itself
supplies (`SurfaceTemperature` from `meteo["temperature"]`) is used instead of the default
here, and any of the six is overridable through `options` (e.g.
`options={"SurfacePressure": 100000.0}`) -- an override wins even over a value the case
itself supplies. The defaults are the values the paper's verified MUNICH runs used."""

_DERIVABLE_INTER_FIELDS = ("WindDirection", "WindSpeed", "PBLH", "UST", "LMO")
"""The five `...Inter` junction meteo fields MUNICH v2.2 requires whenever `With_transport:
yes` (this writer's own default) -- `StreetNetworkTransport.cxx:1690-1699`'s "is needed but
no input data file was provided" checks. When `meteo_junction` supplies none of these,
`write_munich_case` derives it from the streets meeting at that junction (in
`network.junctions` order), with the same reduction `case.py`'s network-wide ("uniform")
meteo uses: `circular_mean_rad` for `WindDirection`, `reciprocal_mean` for `LMO`, a plain
mean otherwise."""

_REQUIRED_TRANSPORT_METEO = ("h_abl", "u_star", "lmo")
"""Street-level `meteo` keys `write_munich_case` itself requires -- MUNICH's own `PBLH`,
`UST`, `LMO` (`_METEO_FIELDS`) are, like their `...Inter` counterparts
(`_DERIVABLE_INTER_FIELDS`), needed whenever `With_transport: yes` (this writer's own
default, always set): `StreetNetworkTransport.cxx:1690-1699`'s "is needed but no input data
file was provided" checks. Unlike the six `_REQUIRED_METEO_DEFAULTS` fields, MUNICH has no
built-in default for these three, and a street-level field that is entirely absent cannot be
derived into its `...Inter` form either, so `write_munich_case` refuses up front instead of
writing a case MUNICH itself would refuse to run."""


def _lonlat(x_m: float, y_m: float, *, lat_ref_deg: float, lat0_deg: float,
            lon0_deg: float) -> tuple[float, float]:
    """The inverse of the equirectangular projection `x = R cos(lat_ref) (lon - lon0)`,
    `y = R (lat - lat0)` -- `read_munich_case`'s own with `lat0 = lon0 = 0`."""
    lon = lon0_deg + math.degrees(x_m / (EARTH_RADIUS_M * math.cos(math.radians(lat_ref_deg))))
    lat = lat0_deg + math.degrees(y_m / EARTH_RADIUS_M)
    return lon, lat


def _is_constant(arr: np.ndarray) -> bool:
    return bool(np.all(arr == arr.flat[0]))


def circular_mean_rad(rad: np.ndarray, axis: int) -> np.ndarray:
    """The mean DIRECTION of angles in radians -- invariant to where the angles wrap (an
    arithmetic mean of angles straddling the wrap lands on the opposite side). Shared by
    `case.py`'s network-wide ("uniform") meteo reduction and this module's derivation of a
    junction's meteo from the streets that meet there, when a case supplies no
    `meteo_junction` of its own -- see `_DERIVABLE_INTER_FIELDS`."""
    return np.arctan2(np.sin(rad).mean(axis=axis), np.cos(rad).mean(axis=axis)) % (2.0 * np.pi)


def reciprocal_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """The mean of a quantity through its reciprocal, `1 / mean(1 / x)`: the Obukhov length's
    stability branches depend continuously on `1/L`, and a plain mean of `L` across values
    that straddle zero (stable next to unstable) can land on the wrong sign. Shared the same
    way as `circular_mean_rad`."""
    return 1.0 / np.mean(1.0 / values, axis=axis)


def write_munich_case(
    out_dir: Path,
    *,
    network: StreetNetwork,
    times: Sequence[float],
    start: datetime,
    junction_ids: Sequence[str],
    species: Sequence[str],
    meteo: Mapping[str, np.ndarray],
    meteo_junction: Mapping[str, np.ndarray],
    emissions: np.ndarray,
    background: np.ndarray,
    native: Mapping[str, Mapping] | None = None,
    options: Mapping[str, object] | None = None,
) -> Path:
    """Writes a complete MUNICH case: `munich.cfg`, `munich-data.cfg`, `munich-saver.cfg`,
    `species.dat`, `street.dat`, `intersection.dat`, and whichever binaries the arrays need.
    Returns `out_dir`.

    Every input is in the neutral vocabulary `StreetCase` uses: `meteo` holds
    `(n_hours, n_streets)` arrays under `wind_dir_from_deg`, `wind_speed`, `h_abl`,
    `u_star`, `lmo`, `temperature`; `meteo_junction` the same keys `(n_hours, n_junctions)`
    in `network.junctions` order (written as MUNICH's `...Inter` fields, in the order
    `intersection.dat`'s rows are written); `emissions` `(n_hours, n_streets, n_species)`
    kg/s; `background` the same shape in kg/m3. The direction is converted to MUNICH's
    radians TOWARD, and masses to micrograms, here. An array holding one value throughout
    is written as an `is_num` constant; any other as a float32 binary. `meteo` must have
    `h_abl`, `u_star` and `lmo` (see `_REQUIRED_TRANSPORT_METEO`): MUNICH needs their
    `PBLH`, `UST`, `LMO` fields, and the `...Inter` counterparts derived from them, whenever
    `With_transport: yes` (always set here), and a `ValueError` names whichever is missing
    up front rather than writing a case MUNICH itself would refuse to run.

    `junction_ids` (in `network.junctions` order) become `intersection.dat`'s ids when every
    one is a distinct whole number, as MUNICH's are (so a read case writes back with its own
    ids); otherwise the junctions are numbered 1, 2, ... in that order.

    `times` must be evenly spaced (it sets `Delta_t`; one time step writes 3600 s); `start`
    becomes every section's `Date_min`.

    Every meteo binary is written with 2 look-ahead records (the last row repeated):
    MUNICH's saver reads up to two steps past the run's last hour to finish its hourly
    averaging, when every meteo field varies in time. Emission and background need none.

    The `[meteo]` `Fields` list always carries the six MUNICH v2.2 requires regardless of
    the case's own meteo (see `_REQUIRED_METEO_DEFAULTS`): `Rain`, `SolarRadiation`,
    `SpecificHumidity`, `SurfacePressure`, `SurfaceTemperature`, `Attenuation` -- from the
    case where it has one (only `SurfaceTemperature`, from `meteo["temperature"]`), else
    `_REQUIRED_METEO_DEFAULTS`'s constant; `options` overrides any of the six, WINNING even
    over a value the case itself supplies.

    Likewise, any of the five `...Inter` junction fields `meteo_junction` does not supply
    (see `_DERIVABLE_INTER_FIELDS`) is derived from the streets meeting at that junction,
    when the case has the corresponding street-level field.

    `native`, when given (a case read from MUNICH files), supplies its own `[street]`
    section (over `_DEFAULT_STREET_OPTIONS`, file names excepted) and its own `projection`.
    `options` overrides any `[street]` key or `_REQUIRED_METEO_DEFAULTS` key (a non-numeric
    value for the latter raises `ValueError` naming the key), and two further keys set the
    geographic anchor of a network in local metres: `lat0_deg`, `lon0_deg` (the lon/lat of
    `(x, y) = (0, 0)`; default `DEFAULT_LAT0_DEG`, `DEFAULT_LON0_DEG`).
    """
    out_dir = Path(out_dir)
    native = dict(native or {})
    options = dict(options or {})
    meteo_overrides: dict[str, float] = {}
    for key in list(options):
        if key not in _REQUIRED_METEO_DEFAULTS:
            continue
        raw = options.pop(key)
        try:
            meteo_overrides[key] = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"write_case: options[{key!r}] = {raw!r} is not a number"
            ) from None
    species = list(species)
    n_hours = len(times)
    n_streets = len(network.streets)
    junctions = network.junctions
    n_junctions = len(junctions)

    def check_shape(label: str, value, shape: tuple[int, ...]) -> None:
        if np.shape(value) != shape:
            raise ValueError(
                f"write_case: {label} must have shape {shape}, got {np.shape(value)}"
            )

    for key, value in meteo.items():
        check_shape(f"meteo[{key!r}]", value, (n_hours, n_streets))
    for key, value in meteo_junction.items():
        check_shape(f"meteo_junction[{key!r}]", value, (n_hours, n_junctions))
    check_shape("emissions", emissions, (n_hours, n_streets, len(species)))
    check_shape("background", background, (n_hours, n_streets, len(species)))

    missing_transport = [k for k in _REQUIRED_TRANSPORT_METEO if k not in meteo]
    if missing_transport:
        raise ValueError(
            f"write_case: format='munich' needs meteo keys {list(_REQUIRED_TRANSPORT_METEO)} "
            f"(each (n_hours, n_streets)); missing {missing_transport}"
        )

    if n_hours > 1:
        steps = np.diff(np.asarray(times, dtype=np.float64))
        if not np.allclose(steps, steps[0], rtol=0, atol=1e-9) or steps[0] <= 0:
            raise ValueError(
                f"write_case: MUNICH needs evenly spaced, increasing times (one Delta_t); "
                f"got steps {steps.tolist()}"
            )
        delta_t = float(steps[0])
    else:
        delta_t = 3600.0

    projection = dict(native.get("projection", {}))
    if "lat0_deg" in options or "lon0_deg" in options or not projection:
        lat0 = float(options.pop("lat0_deg", DEFAULT_LAT0_DEG))
        lon0 = float(options.pop("lon0_deg", DEFAULT_LON0_DEG))
        projection = {"lat_ref_deg": lat0, "lat0_deg": lat0, "lon0_deg": lon0}

    street_options = dict(_DEFAULT_STREET_OPTIONS)
    street_options.update({k: str(v) for k, v in native.get("street", {}).items()
                           if k not in _FILE_KEYS})
    street_options.update({k: str(v) for k, v in options.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)

    ids = [str(j) for j in junction_ids]
    if not (len(ids) == n_junctions and all(j.isdigit() for j in ids)
            and len(set(ids)) == n_junctions):
        ids = [str(i + 1) for i in range(n_junctions)]
    junction_id = dict(zip(junctions, ids, strict=True))
    street_touches: dict[str, list[str]] = {name: [] for name in junctions}
    for street in network.streets:
        street_touches[street.u].append(street.name)
        street_touches[street.v].append(street.name)

    intersection_lines = ["#id;lon;lat;number_of_streets;1st_street_id;2nd_street_id;..."]
    for name in junctions:
        lon, lat = _lonlat(network.x[name], network.y[name], **projection)
        touching = street_touches[name]
        row = [junction_id[name], repr(lon), repr(lat), str(len(touching)), *touching]
        intersection_lines.append(";".join(row) + ";")
    (out_dir / "intersection.dat").write_text("\n".join(intersection_lines) + "\n")

    street_lines = ["#" + ";".join(_STREET_COLUMNS)]
    for street in network.streets:
        row = [
            street.name, junction_id[street.u], junction_id[street.v],
            repr(street.length), repr(street.width), repr(street.height), "0",
        ]
        street_lines.append(";".join(row))
    (out_dir / "street.dat").write_text("\n".join(street_lines) + "\n")

    # [reactivity]/[alpha]/[beta]/[henry]/[diffusivity] are read unconditionally by
    # `StreetNetworkTransport::ReadConfiguration` regardless of `With_deposition` -- their
    # values are unused with deposition off (this writer's `[options]` block always turns it
    # off), so any harmless placeholder value works for a generic species name.
    def species_section(title: str, value: str, comment: str | None = None) -> list[str]:
        lines = [f"[{title}]", ""]
        if comment is not None:
            lines += [comment, ""]
        lines += [f"{sp} {value}" for sp in species]
        return lines

    species_lines = ["[species]", "", " ".join(species), ""]
    species_lines += species_section("molecular_weight", "46.", "# Unit: g / mol.") + [""]
    species_lines += species_section("reactivity", "0.1") + [""]
    species_lines += species_section("alpha", "0.") + [""]
    species_lines += species_section("beta", "0.8") + [""]
    species_lines += species_section("henry", "1.2e-2") + [""]
    species_lines += species_section("diffusivity", "0.14")
    (out_dir / "species.dat").write_text("\n".join(species_lines) + "\n")

    street_block = "\n".join(f"{k}: {v}" for k, v in street_options.items())
    date_min = format_date(start)

    munich_cfg = f"""\
[display]

Show_iterations: yes
Show_date: yes
Show_configuration: yes

[domain]

Date_min: {date_min}
Delta_t: {delta_t!r}
Nt: {n_hours}
Species: species.dat

[data]

Data_description: munich-data.cfg

[output]

Configuration_file: munich-saver.cfg

[options]

With_chemistry: no
Option_chemistry: Leighton
With_photolysis: no
With_deposition: no
With_scavenging: no
With_transport: yes

[street]

{street_block}
"""
    (out_dir / "munich.cfg").write_text(munich_cfg)

    saver_cfg = """\
Result_dir: results/

[save]

Species: all

Date_beg: -1
Date_end: -1
Interval_length: 1
Averaged: yes
Initial_concentration: no

Type: street

Output_file: <Result_dir>/&f.bin

Text_file: no
"""
    (out_dir / "munich-saver.cfg").write_text(saver_cfg)

    def section_head(title: str, nt: int, fields: Sequence[str]) -> list[str]:
        return [f"[{title}]", "", f"Date_min: {date_min}", f"Delta_t: {delta_t!r}",
                f"Nt: {nt}", f"Fields: {' '.join(fields)}", "Filename: 0.0"]

    def mass_section(title: str, values: np.ndarray, stem: str) -> list[str]:
        lines = section_head(title, n_hours, species)
        for s, sp in enumerate(species):
            arr = np.asarray(values[:, :, s], dtype=np.float64) * UG_PER_KG
            if _is_constant(arr):
                lines.append(f"{sp} {float(arr.flat[0])!r}")
            else:
                filename = f"{stem}_{sp}.bin"
                _write_binary(out_dir / filename, arr)
                lines.append(f"{sp} {filename}")
        return lines

    data_lines = mass_section("emission", emissions, "emission")
    data_lines += [""] + mass_section("background_concentration", background, "background")

    to_munich = {key: name for name, key in _METEO_FIELDS.items()}
    to_munich["wind_dir_from_deg"] = "WindDirection"
    entries: list[tuple[str, np.ndarray]] = []
    for table, suffix in ((meteo, ""), (meteo_junction, _INTER_SUFFIX)):
        for key, value in table.items():
            if key not in to_munich:
                raise ValueError(
                    f"write_case: meteo key {key!r} has no MUNICH field; expected one of "
                    f"{sorted(to_munich)}"
                )
            arr = np.asarray(value, dtype=np.float64)
            if key == "wind_dir_from_deg":
                arr = deg_from_to_munich_rad(arr)
            entries.append((to_munich[key] + suffix, arr))

    # MUNICH needs every `...Inter` junction field whenever `With_transport: yes` (see
    # `_DERIVABLE_INTER_FIELDS`); derive whichever one `meteo_junction` did not supply from
    # the streets meeting at that junction, when the street-level field itself is present.
    street_col = {street.name: i for i, street in enumerate(network.streets)}
    by_name = {name: arr for name, arr in entries}
    present_inter = {name for name in by_name if name.endswith(_INTER_SUFFIX)}
    for name in _DERIVABLE_INTER_FIELDS:
        inter_name = name + _INTER_SUFFIX
        if inter_name in present_inter or name not in by_name:
            continue
        arr = by_name[name]                                        # (n_hours, n_streets)
        columns = []
        for junction in junctions:
            cols = [street_col[s] for s in street_touches[junction]]
            sub = arr[:, cols]
            if name == "WindDirection":
                columns.append(circular_mean_rad(sub, axis=1))
            elif name == "LMO":
                columns.append(reciprocal_mean(sub, axis=1))
            else:
                columns.append(sub.mean(axis=1))
        entries.append((inter_name, np.stack(columns, axis=1)))

    # An `options` override of one of the six always-required fields wins even over a value
    # the case itself supplies (e.g. `meteo["temperature"]`'s own `SurfaceTemperature`).
    entries = [(name, arr) for name, arr in entries if name not in meteo_overrides]
    present = {name for name, _ in entries}
    for name, default in _REQUIRED_METEO_DEFAULTS.items():
        if name in meteo_overrides:
            entries.append((name, np.array([meteo_overrides[name]])))
        elif name not in present:
            entries.append((name, np.array([default])))

    n_meteo_rows = n_hours + 2
    meteo_lines = [""] + section_head("meteo", n_meteo_rows, [name for name, _ in entries])
    for name, arr in entries:
        if _is_constant(arr):
            meteo_lines.append(f"{name} {float(arr.flat[0])!r}")
        else:
            pad = np.repeat(arr[-1:], n_meteo_rows - n_hours, axis=0)
            _write_binary(out_dir / f"meteo_{name}.bin", np.concatenate([arr, pad]))
            meteo_lines.append(f"{name} meteo_{name}.bin")
    data_lines += meteo_lines
    (out_dir / "munich-data.cfg").write_text("\n".join(data_lines) + "\n")

    return out_dir
