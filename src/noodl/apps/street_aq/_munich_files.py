"""MUNICH's own street-network file format: `munich.cfg` plus the street/intersection
tables, the `[section]`/`key: value` config dialect, and the `float32` binaries it names.

Private to `noodl.apps.street_aq`: `case.py` is the format-neutral public surface
(`StreetCase`, `read_case`, `write_case`, `drivers_at`); this module knows nothing about
that dataclass -- it reads and writes plain dicts/arrays, so it stays reusable for a future
SIRANE module without either importing the other.

The schema here is the one discovered by reading the MUNICH v2.2 source
(`cerea-lab/munich`; the pinned commit is recorded in `noodl-paper/paper/munich/README.md`)
and its shipped `processing/photochemistry` example, NOT the full Talos config grammar --
this module reads and writes exactly what `write_munich_case` produces (one `key: value`
or `key = value` pair per line, or a bare `key value` override line, inside `[section]`
blocks), which is also what MUNICH's own `Talos::ConfigStream` accepts (verified in
`noodl-paper` by running the real `munich` binary on this writer's output).

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
from pathlib import Path

import numpy as np

from noodl.apps.street_aq.network import Street, StreetNetwork

EARTH_RADIUS_M = 6371229.0
"""`StreetNetworkTransport.cxx:18`'s `earth_radius` -- the constant MUNICH itself uses to
turn a street's endpoint lon/lat into metres. Reusing it here means a network written by
`write_munich_case` and re-projected internally by MUNICH recovers the same `x`, `y` metres
this reader would compute directly, to sub-millimetre precision on a few-hundred-metre
network (`read_munich_case` re-derives its own reference latitude from the file's own
coordinates, not from whatever `lat0_deg` the writer used, but the two differ only by the
network's own north-south extent divided by the Earth's radius -- a few parts per million on
a network a few hundred metres across)."""

UG_PER_KG = 1e9
"""MUNICH's native mass unit (micrograms) per kg (noodl's SI unit). `read_munich_case`
divides emissions/background by this AT READ TIME; `write_munich_case` multiplies by it at
write time. The one factor is defined once, here, where MUNICH's units are documented, not
duplicated as a second magic number anywhere else."""

_METEO_FIELDS = {
    "WindSpeed": "wind_speed",
    "PBLH": "h_abl",
    "UST": "u_star",
    "LMO": "lmo",
    "SurfaceTemperature": "temperature",
}
"""MUNICH meteo field name -> the `StreetCase.meteo` key it becomes. `WindDirection` is
handled separately because it needs the degrees-FROM conversion (`munich_rad_to_deg_from`)."""

_INTER_SUFFIX = "Inter"


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


def _resolve_field(
    section: Mapping[str, str], field: str, *, root: Path, nt: int, n_columns: int
) -> np.ndarray:
    """`section`'s value for `field` (an override line, or the section's generic
    `Filename` template with `&f` replaced) -- a constant broadcast to `(nt, n_columns)`
    if it parses as a number (MUNICH's own `InputFiles::Read`/`is_num` shortcut), else a
    `(nt, n_columns)` float32 binary file read relative to `root`."""
    raw = section.get(field)
    if raw is None:
        raw = section["Filename"].replace("&f", field)
    if _is_num(raw):
        return np.full((nt, n_columns), float(raw), dtype=np.float64)
    return _read_binary_field(root / raw, nt, n_columns)


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


def _truncate_or_tile(field: np.ndarray, nt: int) -> np.ndarray:
    """`field[:nt]` if it has at least `nt` rows, else `field` tiled to at least `nt` rows
    and truncated -- the same look-ahead handling MUNICH's own saver needs on the write side
    (see `write_munich_case`'s `n_hours + 2`), inverted for reading."""
    if field.shape[0] >= nt:
        return field[:nt]
    reps = -(-nt // field.shape[0])  # ceil division
    return np.tile(field, (reps, 1))[:nt]


# ------------------------------------------------------------------------------- reading

def read_munich_case(root: Path) -> dict:
    """Every array here is already SI (converting MUNICH's own micrograms at read time --
    see `UG_PER_KG`), and every per-street array is kept PER STREET (not reduced to a
    network-wide value the way an earlier version of this reader did) -- reduction to
    whatever a particular `Model` needs is `noodl.apps.street_aq.case.drivers_at`'s job, not
    the reader's, so the same `StreetCase` can drive a uniform or a per-street model.

    Returns a plain dict of the fields `StreetCase` takes: `network`, `times`, `street_ids`,
    `junction_ids` (in `network.junctions` order), `species`, `meteo`, `meteo_junction`,
    `emissions`, `background`, `native`.
    """
    root = Path(root)
    cfg = _parse_cfg(root / "munich.cfg")
    domain = cfg["domain"]
    nt = int(float(domain["Nt"]))
    delta_t = float(domain["Delta_t"])
    times = [i * delta_t for i in range(nt)]

    species = _read_species_list(root / domain["Species"])

    street_section = cfg["street"]
    street_rows = _read_semicolon_table(root / street_section["Street"])
    intersection_rows = _read_semicolon_table(root / street_section["Intersection"])

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

    data_cfg = _parse_cfg(root / cfg["data"]["Data_description"])

    emission_section = data_cfg["emission"]
    n_emission_t = int(float(emission_section["Nt"]))
    emissions = np.zeros((nt, n_streets, len(species)), dtype=np.float64)
    for s, sp in enumerate(species):
        field = _resolve_field(
            emission_section, sp, root=root, nt=n_emission_t, n_columns=n_streets
        )
        emissions[:, :, s] = _truncate_or_tile(field, nt)
    emissions /= UG_PER_KG  # MUNICH's micrograms/s -> kg/s.

    background_section = data_cfg["background_concentration"]
    n_bg_t = int(float(background_section["Nt"]))
    background = np.zeros((nt, n_streets, len(species)), dtype=np.float64)
    for s, sp in enumerate(species):
        # Per-street, exactly like emission and meteo (`StreetNetworkTransport.cxx:600`,
        # `Background_i.Resize(GridS2D, GridST2D)`) -- kept per-street here, not reduced to
        # a domain mean the way an earlier version of this reader did.
        field = _resolve_field(
            background_section, sp, root=root, nt=n_bg_t, n_columns=n_streets
        )
        background[:, :, s] = _truncate_or_tile(field, nt)
    background /= UG_PER_KG  # MUNICH's micrograms/m3 -> kg/m3.

    meteo_section = data_cfg["meteo"]
    n_meteo_t = int(float(meteo_section["Nt"]))
    fields_available = set(meteo_section.get("Fields", "").split())

    def street_field(name: str) -> np.ndarray:
        field = _resolve_field(
            meteo_section, name, root=root, nt=n_meteo_t, n_columns=n_streets
        )
        return _truncate_or_tile(field, nt)

    def junction_field(name: str) -> np.ndarray:
        field = _resolve_field(
            meteo_section, name, root=root, nt=n_meteo_t, n_columns=n_intersections
        )
        return _truncate_or_tile(field, nt)[:, junction_cols]

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

    native = {name: dict(values) for name, values in cfg.items()}

    return dict(
        network=network, times=times, street_ids=street_ids, junction_ids=junction_ids,
        species=species, meteo=meteo, meteo_junction=meteo_junction,
        emissions=emissions, background=background, native=native,
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
iteration, not MUNICH's default unstable explicit integrator -- see
`noodl-paper/paper/munich/README.md`, "MUNICH's time integration") -- plus the keys MUNICH
v2.2 requires regardless of the options actually exercised (`Compute_Macdonald_from`,
`Deposition_wind_profile`, `Sub_delta_t_min`, `Building_density`, `With_tree_aerodynamic`,
`With_tree_deposition`; the shipped example predates them and was never run to notice). Any
key here may be overridden through `write_munich_case`'s own `options` argument."""


def _broadcast_to_hours_columns(value, n_hours: int, n_columns: int) -> np.ndarray:
    """A meteo/emission/background array in `(n_hours, n_columns)` layout: a `(n_hours,)`
    array gains a column axis (the same value in every column); a `(n_hours, n_columns)`
    array is passed through, checked. `n_columns` is `len(network.streets)` for a
    street-indexed field, or `len(network.junctions)` for one of MUNICH's `...Inter`
    (per-intersection) fields -- the caller picks."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != n_hours:
            raise ValueError(
                f"write_case: a 1-D array must have length n_hours={n_hours}, got shape "
                f"{arr.shape}"
            )
        return np.tile(arr.reshape(n_hours, 1), (1, n_columns))
    if arr.ndim == 2:
        if arr.shape != (n_hours, n_columns):
            raise ValueError(
                f"write_case: a 2-D array must have shape (n_hours, n_columns) = "
                f"({n_hours}, {n_columns}), got {arr.shape}"
            )
        return arr
    raise ValueError(f"write_case: an array driver must be 1-D or 2-D, got ndim={arr.ndim}")


def _lonlat(x_m: float, y_m: float, *, lat0_deg: float, lon0_deg: float) -> tuple[float, float]:
    """The inverse of `read_munich_case`'s projection -- see `EARTH_RADIUS_M`."""
    lat0_rad = math.radians(lat0_deg)
    lon = lon0_deg + math.degrees(x_m / (EARTH_RADIUS_M * math.cos(lat0_rad)))
    lat = lat0_deg + math.degrees(y_m / EARTH_RADIUS_M)
    return lon, lat


def write_munich_case(
    out_dir: Path,
    network: StreetNetwork,
    *,
    species: Sequence[str],
    date_min: str,
    n_hours: int,
    meteo: Mapping[str, float | np.ndarray],
    emissions_kg_s: float | np.ndarray,
    background_kg_m3: float | np.ndarray,
    options: Mapping[str, str] | None = None,
    lat0_deg: float = 48.85,
    lon0_deg: float = 2.35,
) -> Path:
    """Writes a complete MUNICH case for `network`: `munich.cfg`, `munich-data.cfg`,
    `munich-saver.cfg`, `species.dat`, `street.dat`, `intersection.dat`, and whichever
    binaries the array-valued `meteo`/`emissions_kg_s`/`background_kg_m3` need. Returns
    `out_dir`.

    `meteo` keys are MUNICH's own field names (`WindDirection`, `WindSpeed`, `PBLH`, `UST`,
    `LMO`, `SurfaceTemperature`, or their `...Inter` per-intersection counterparts); a scalar
    value is written as an `is_num` constant override, an array as a `(n_hours,)` (broadcast
    across columns) or `(n_hours, n_columns)` binary -- `n_columns = len(network.streets)`
    for a plain field, `len(network.junctions)` for an `...Inter` field, in `network.junctions`
    order (the same order `intersection.dat`'s rows are written in, just above). `WindDirection`
    is MUNICH's own convention (radians TOWARD, clockwise from north) -- this writer does no
    conversion, so a caller wanting a degrees-FROM direction must convert it first with
    `case.deg_from_to_munich_rad`.

    Every array-valued meteo field is written with `n_hours + 2` look-ahead rows (the last
    row repeated): MUNICH's saver reads up to two steps past the run's last hour to finish
    its hourly averaging (`noodl-paper/paper/munich/README.md`, "The real Paris run" --
    the idealised case only needed one look-ahead row, but the real Paris run, with every
    meteo field genuinely time-varying, needed two; this writer always writes two, the safe
    superset). `emissions_kg_s`/`background_kg_m3` need no look-ahead (only meteo does).

    `options` overrides the default `[street]` section (`_DEFAULT_STREET_OPTIONS` --
    MUNICH's equivalents of `canyon_wind="exponential"`, `exchange="schulte"`,
    `roof_wind_form="sirane"`, `direction_averaging="munich"`, and
    `With_stationary_hypothesis: yes`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    species = list(species)
    n_streets = len(network.streets)

    junctions = network.junctions
    n_junctions = len(junctions)
    junction_id = {name: str(i + 1) for i, name in enumerate(junctions)}
    street_touches: dict[str, list[str]] = {name: [] for name in junctions}
    for street in network.streets:
        street_touches[street.u].append(street.name)
        street_touches[street.v].append(street.name)

    intersection_lines = ["#id;lon;lat;number_of_streets;1st_street_id;2nd_street_id;..."]
    for name in junctions:
        lon, lat = _lonlat(network.x[name], network.y[name], lat0_deg=lat0_deg, lon0_deg=lon0_deg)
        touching = street_touches[name]
        row = [junction_id[name], repr(lon), repr(lat), str(len(touching)), *touching]
        intersection_lines.append(";".join(row) + ";")
    (out_dir / "intersection.dat").write_text("\n".join(intersection_lines) + "\n")

    street_lines = ["#id;begin_inter;end_inter;length;width;height;typo"]
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

    street_options = dict(_DEFAULT_STREET_OPTIONS)
    if options:
        street_options.update({k: str(v) for k, v in options.items()})
    street_block = "\n".join(f"{k}: {v}" for k, v in street_options.items())

    munich_cfg = f"""\
[display]

Show_iterations: yes
Show_date: yes
Show_configuration: yes

[domain]

Date_min: {date_min}
Delta_t: 3600.0
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

    def constant_or_array(value) -> tuple[np.ndarray | None, float | None]:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return None, float(arr)
        return _broadcast_to_hours_columns(arr, n_hours, n_streets), None

    def mass_section(title: str, value, *, filename: str) -> list[str]:
        arr, const = constant_or_array(value)
        lines = [
            f"[{title}]", "", f"Date_min: {date_min}", "Delta_t: 3600.0", f"Nt: {n_hours}",
            f"Fields: {' '.join(species)}",
        ]
        if const is not None:
            lines.append(f"Filename: {const * UG_PER_KG!r}")
        else:
            lines.append("Filename: 0.0")
            _write_binary(out_dir / filename, arr * UG_PER_KG)
            lines += [f"{sp} {filename}" for sp in species]
        return lines

    data_lines = mass_section("emission", emissions_kg_s, filename="emission.bin")
    data_lines += [""] + mass_section(
        "background_concentration", background_kg_m3, filename="background.bin"
    )

    n_meteo_rows = n_hours + 2
    meteo_lines = [
        "", "[meteo]", "", f"Date_min: {date_min}", "Delta_t: 3600.0", f"Nt: {n_meteo_rows}",
        f"Fields: {' '.join(meteo.keys())}", "Filename: 0.0",
    ]
    for key, value in meteo.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            meteo_lines.append(f"{key} {float(arr)!r}")
        else:
            n_columns = n_junctions if key.endswith(_INTER_SUFFIX) else n_streets
            full = _broadcast_to_hours_columns(arr, n_hours, n_columns)
            pad = np.tile(full[-1:], (n_meteo_rows - n_hours, 1))
            _write_binary(out_dir / f"meteo_{key}.bin", np.concatenate([full, pad]))
            meteo_lines.append(f"{key} meteo_{key}.bin")
    data_lines += meteo_lines
    (out_dir / "munich-data.cfg").write_text("\n".join(data_lines) + "\n")

    return out_dir
