"""TMY3 weather data as MBL's `ReaderTMY3` puts it on the weather bus, for airflow.

Transcribed from MBL v13.0.0 (commit 55abf579) and MSL 4.1.0 sources, never from their
documentation. Only the five weather-bus outputs an `Outside` boundary reads are evaluated
(`schema.WEATHER_BUS_OUTPUTS`): dry-bulb temperature `TDryBul` (K), atmospheric pressure `pAtm`
(Pa), relative humidity `relHum` (1), wind speed `winSpe` (m/s) and wind direction `winDir`
(rad, the direction the wind comes FROM: N = 0, E = pi/2, as in the TMY3 file). Solar and sky
outputs are not.

File
----
`ReaderTMY3.mo:243-250`: `datRea` is a `CombiTable1Ds` on table `tab1` of the `.mos` file
`filNam`. The outputs read here are the file's columns (1-based) 2 (dry bulb, degC), 4
(relative humidity, %), 5 (station pressure, Pa), 16 (wind direction, deg) and 17 (wind speed,
m/s) (`datRea.y[1]`, `[3]`, `[4]`, `[11]`, `[12]`, `ReaderTMY3.mo:593-681`); column 1 is the
time in seconds from the start of the year. `smoothness = ContinuousDerivative`: MSL's Akima
spline (`ModelicaStandardTables.c`, `akimaSpline1DInit` :6543-6611; linear segments for fewer
than three rows, :2275-2281), extrapolated linearly from the end slopes (`LastTwoPoints`, the
`CombiTable1Ds` default, :2453-2481). The spline runs on the FILE's values; the units are
converted afterwards.

Time
----
The table's abscissa is the calendar time `conTim.calTim` (`ReaderTMY3.mo:370-372, 581`;
`BaseClasses/PartialConvertTime.mo:16-36`). With `lenWea` = last time stamp + the mean increment -
first time stamp (`Resources/C-Sources/getTimeSpan.c:135-142`), a file whose length is a whole
number of 365-day years repeats: `calTim = time - (k - 1) lenWea`, `k = integer(t0/lenWea) + 1`
at the start, incremented each time `time` passes `k lenWea` (a time event, where the outputs
jump: `reader_events`). Any other file is read at `calTim = time` and must cover the run, from
its first time stamp to 1800 s past its end (`ConvertTime.mo:22-32`). The 30-minute shift of
`datRea30Min` applies to the solar columns only and is not used here.

Conversions, limits and checks (applied whatever the source)
-----------------------------------------------------------
`TDryBul = u - T_zero = u + 273.15` (`From_degC`, MSL `Units.mo:1343`), checked in
(203.15, 343.15) K (`CheckDryBulbTemperature.mo`); `pAtm = u`, checked in (3100, 120000) Pa
(`CheckPressure.mo`); `relHum = min(1, max(0, 0.01 u))` (`conRelHum`, `Gain(k = 0.01)`,
`ReaderTMY3.mo:318`; `LimiterRelativeHumidity.mo:10`, `PartialLimiter.mo:6-7`);
`winSpe = max(0, u)` (`LimiterWindSpeed.mo`); `winDir = mod((pi/180) u, 2 pi)` (`From_deg`,
`Units.mo:1429`; `LimiterWindDirection.mo`). A check MBL asserts is a named refusal here.

Sources
-------
Each output's `<output>Sou` (`SourceSelector.mo`): `File` (the default for all but `pAtmSou`,
`ReaderTMY3.mo:18, 34, 79, 95, 109`), `Parameter` (the reader's parameter of the output's
name, defaults `pAtm = 101325`, `TDryBul = 293.15`, `relHum = 0.5`, `winSpe = 1`, `winDir = 1`,
`ReaderTMY3.mo:21, 37, 84, 97, 111`) or `Input` (the signal driving `<reader>.<output>_in`).

The weather file travels with the export: `scripts/modelica_export.py` copies it next to the
JSON and records `"files": {"filNam": {"path", "sha256"}}` (`resolve_file`).
"""

from __future__ import annotations

import functools
import hashlib
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from noodl.apps.building_physics.modelica.schema import (
    WEATHER_BUS_OUTPUTS,
    Component,
    ModelicaDoc,
    ModelicaImportError,
)

Tensor = torch.Tensor
F64 = torch.float64

TABLE = "tab1"  # ReaderTMY3.mo:245
FILE_COLUMN = {"TDryBul": 2, "relHum": 4, "pAtm": 5, "winDir": 16, "winSpe": 17}
DEFAULT_SOURCE = {"TDryBul": "File", "pAtm": "Parameter", "relHum": "File", "winSpe": "File",
                  "winDir": "File"}
DEFAULT_VALUE = {"TDryBul": 293.15, "pAtm": 101325.0, "relHum": 0.5, "winSpe": 1.0,
                 "winDir": 1.0}
# Where a limiter cuts a File output, in the FILE's units (`reader_events`): relative humidity
# at 0 and 100 %, wind speed at 0 m/s.
LIMIT_LEVELS = {"relHum": (0.0, 100.0), "winSpe": (0.0,)}
YEAR = 365 * 24 * 3600.0  # PartialConvertTime.mo:16
SOLAR_SHIFT = 1800.0  # ConvertTime.mo:15
T_BOUNDS = (203.15, 343.15)  # K, CheckDryBulbTemperature.mo
P_BOUNDS = (3100.0, 120000.0)  # Pa, CheckPressure.mo
_HEADER = re.compile(r"^\s*(?:double|float)\s+(\w+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_SEPARATORS = re.compile(r"[\s,;]+")


class AkimaSpline:
    """MSL's `ContinuousDerivative` interpolation of `y` over strictly increasing `x`
    (`ModelicaStandardTables.c`: `akimaSpline1DInit` :6543-6611, `CombiTable1D` evaluation
    with `LastTwoPoints` extrapolation :2433-2481; linear for two points, :2275-2281).

    Divided differences `d[2..n]`, extended by `d[0] = 3 d[2] - 2 d[3]`, `d[1] = 2 d[2] - d[3]`,
    `d[n+1] = 2 d[n] - d[n-1]`, `d[n+2] = 3 d[n] - 2 d[n-1]`; the slope at knot `j` is
    `(1 - a) d[j+1] + a d[j+2]` with `a = |d[j+1] - d[j]| / (|d[j+3] - d[j+2]| + |d[j+1] - d[j]|)`
    (the mean of `d[j+1]` and `d[j+2]` when both weights vanish); each interval is the cubic
    Hermite through its knots with those slopes."""

    def __init__(self, x: Sequence[float], y: Sequence[float]) -> None:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = x.size
        if n < 2 or y.size != n:
            raise ValueError("AkimaSpline: needs at least two points and one y per x")
        if not bool(np.all(np.diff(x) > 0)):
            raise ValueError("AkimaSpline: x must be strictly increasing")
        self.x, self.y = x, y
        dx = np.diff(x)
        if n == 2:
            self.c0 = np.zeros(1)
            self.c1 = np.zeros(1)
            self.c2 = np.array([(y[1] - y[0]) / dx[0]])
            return
        d = np.empty(n + 3)
        d[2:n + 1] = np.diff(y) / dx                       # :6574-6578
        d[0] = 3 * d[2] - 2 * d[3]                         # :6581-6584
        d[1] = 2 * d[2] - d[3]
        d[n + 1] = 2 * d[n] - d[n - 1]
        d[n + 2] = 3 * d[n] - 2 * d[n - 1]
        j = np.arange(n)
        hi = np.abs(d[j + 3] - d[j + 2])
        lo = np.abs(d[j + 1] - d[j])
        s = hi + lo
        a = lo / np.where(s > 0, s, 1.0)
        t = np.where(s > 0, (1 - a) * d[j + 1] + a * d[j + 2],
                     0.5 * d[j + 1] + 0.5 * d[j + 2])  # :6587-6608
        i = np.arange(n - 1)
        self.c2 = t[:-1]
        self.c1 = (3 * d[i + 2] - 2 * t[:-1] - t[1:]) / dx
        self.c0 = (t[:-1] + t[1:] - 2 * d[i + 2]) / (dx * dx)

    def __call__(self, u) -> Tensor:
        u = torch.as_tensor(u, dtype=F64)
        x, y = torch.from_numpy(self.x), torch.from_numpy(self.y)
        c0, c1, c2 = (torch.from_numpy(c) for c in (self.c0, self.c1, self.c2))
        m = x.numel() - 1
        i = (torch.searchsorted(x, u.contiguous(), right=True) - 1).clamp(0, m - 1)
        v = u - x[i]
        inside = y[i] + ((c0[i] * v + c1[i]) * v + c2[i]) * v  # :2440-2442
        left = y[0] + c2[0] * (u - x[0])
        vr = x[-1] - x[-2]
        right = y[-1] + ((3 * c0[-1] * vr + 2 * c1[-1]) * vr + c2[-1]) * (u - x[-1])
        return torch.where(u < x[0], left, torch.where(u > x[-1], right, inside))

    def crossings(self, level: float, lo: float, hi: float) -> list[float]:
        """The points of the open interval `(lo, hi)` where the spline equals `level`, knots
        excluded (a knot is a step boundary already): the real roots of each interval's cubic
        strictly inside it, and the roots of the linear extrapolations."""
        x, y = self.x, self.y
        out: list[float] = []
        first = max(0, int(np.searchsorted(x, lo, side="right")) - 1)
        last = min(x.size - 2, int(np.searchsorted(x, hi, side="left")))
        for i in range(first, last + 1):
            h = x[i + 1] - x[i]
            for r in np.roots([self.c0[i], self.c1[i], self.c2[i], y[i] - level]):
                if abs(r.imag) <= 1e-9 * h and 1e-9 * h < r.real < (1 - 1e-9) * h:
                    out.append(float(x[i] + r.real))
        if lo < x[0] and self.c2[0] != 0:
            u = float(x[0] + (level - y[0]) / self.c2[0])
            if u < x[0]:
                out.append(u)
        if hi > x[-1]:
            vr = x[-1] - x[-2]
            slope = (3 * self.c0[-1] * vr + 2 * self.c1[-1]) * vr + self.c2[-1]
            if slope != 0:
                u = float(x[-1] + (level - y[-1]) / slope)
                if u > x[-1]:
                    out.append(u)

        def near_knot(u: float) -> bool:
            return bool(np.any(np.abs(x - u) <= 1e-9 * max(1.0, abs(u))))

        return sorted(u for u in out if lo < u < hi and not near_knot(u))


class WeatherTable:
    """Table `tab1` of a TMY3 `.mos` file: `data[:, 0]` the time (s), `data[:, k - 1]` the
    file's column `k`."""

    def __init__(self, path: Path, data: np.ndarray) -> None:
        self.path, self.data = path, data
        self._splines: dict[int, AkimaSpline] = {}

    @property
    def time(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def span(self) -> tuple[float, float]:
        """`(first time stamp, last time stamp + mean increment)` (s), getTimeSpan.c:135-142."""
        t = self.time
        interval = (t[-1] - t[0]) / (t.size - 1)
        return float(t[0]), float(t[-1] + interval)

    def spline(self, column: int) -> AkimaSpline:
        """The interpolation of the file's column `column` (1-based) over the time column."""
        if column not in self._splines:
            self._splines[column] = AkimaSpline(self.time, self.data[:, column - 1])
        return self._splines[column]


def read_table(path: str | Path) -> WeatherTable:
    """Table `tab1` of the MSL text-table file `path` (`#1` format: a `double tab1(rows,
    columns)` line, then the rows; `#` lines are comments). Raises `ModelicaImportError` naming
    the file. Cached while the file is unchanged."""
    p = Path(path).resolve()
    st = p.stat()
    return _read_table(str(p), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=8)
def _read_table(path: str, mtime_ns: int, size: int) -> WeatherTable:
    lines = Path(path).read_text().splitlines()
    for k, line in enumerate(lines):
        m = _HEADER.match(line)
        if m is None or m.group(1) != TABLE:
            continue
        rows, cols = int(m.group(2)), int(m.group(3))
        values: list[float] = []
        for raw in lines[k + 1:]:
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            if _HEADER.match(text) or len(values) >= rows * cols:
                break
            try:
                values.extend(float(v) for v in _SEPARATORS.split(text) if v)
            except ValueError as exc:
                raise ModelicaImportError(f"modelica: {path}: table {TABLE}: {exc}") from None
        if len(values) != rows * cols:
            raise ModelicaImportError(
                f"modelica: {path}: table {TABLE} declares {rows} x {cols} values, the file "
                f"holds {len(values)}"
            )
        data = np.asarray(values, dtype=np.float64).reshape(rows, cols)
        need = max(FILE_COLUMN.values())
        if rows < 2 or cols < need:
            raise ModelicaImportError(
                f"modelica: {path}: table {TABLE} is {rows} x {cols}; a TMY3 table needs at "
                f"least 2 rows and {need} columns"
            )
        if not bool(np.all(np.diff(data[:, 0]) > 0)):
            raise ModelicaImportError(
                f"modelica: {path}: table {TABLE}'s time column is not strictly increasing"
            )
        return WeatherTable(Path(path), data)
    raise ModelicaImportError(
        f"modelica: {path}: no table {TABLE!r} (a 'double {TABLE}(rows,columns)' line)"
    )


@functools.lru_cache(maxsize=8)
def _sha256(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_file(comp: Component, doc: ModelicaDoc) -> Path:
    """The weather file of reader `comp`: `files.filNam.path` next to the JSON, with the
    sha256 the export recorded. Raises `ModelicaImportError` naming the reader otherwise."""
    where = f"modelica: {comp.name} ({comp.cls})"
    entry = comp.files.get("filNam")
    if entry is None:
        raise ModelicaImportError(
            f"{where}: its weather file ({comp.parameters.get('filNam', 'filNam not exported')})"
            f" does not travel with the model (the JSON records no files.filNam); re-export "
            f"the model with scripts/modelica_export.py, which copies the file next to the JSON"
        )
    name = entry["path"]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ModelicaImportError(
            f"{where}: weather file {name!r} must be a bare file name in the JSON's directory"
        )
    if doc.source is None:
        raise ModelicaImportError(
            f"{where}: the model was not read from a file, so its weather file {name!r} has "
            f"no directory to be found in"
        )
    path = doc.source.parent / name
    if not path.is_file():
        raise ModelicaImportError(f"{where}: weather file {path} not found")
    st = path.stat()
    digest = _sha256(str(path.resolve()), st.st_mtime_ns, st.st_size)
    if digest != entry["sha256"]:
        raise ModelicaImportError(
            f"{where}: weather file {path} has sha256 {digest}, not the {entry['sha256']} the "
            f"export recorded: it is not the file the model was exported with (a line-ending "
            f"conversion changes it too)"
        )
    return path


def _repeats(length: float) -> bool:
    """`canRepeatWeatherFile` (PartialConvertTime.mo): `abs(mod(lenWea, 365 d)) < 1e-2`."""
    return abs(length - math.floor(length / YEAR) * YEAR) < 1e-2


def calendar_time(t, t_start: float, span: tuple[float, float]) -> Tensor:
    """`conTim.calTim` (PartialConvertTime.mo) at model times `t` (s) of a run starting at
    `t_start` (s), for a file covering `span` (`WeatherTable.span`): `t - tNext + lenWea`,
    `tNext = k lenWea`, evaluated in MBL's order (PartialConvertTime.mo:25-36), so that it
    rounds as MBL's does (to the spacing of doubles near `tNext`)."""
    t = torch.as_tensor(t, dtype=F64)
    first, end = span
    length = end - first
    if not _repeats(length):
        return t.clone()
    k0 = math.floor(t_start / length) + 1
    k = torch.clamp(torch.ceil(t / length), min=k0)
    return (t - k * length) + length


def _short(value, default: str) -> str:
    return str(value if value is not None else default).rsplit(".", 1)[-1]


def source(comp: Component, output: str) -> str:
    """`<output>Sou` of reader `comp`: `"File"`, `"Parameter"` or `"Input"` (another value is
    returned as is and refused by `reader_outputs`)."""
    return _short(comp.parameters.get(f"{output}Sou"), DEFAULT_SOURCE[output])


def _convert(output: str, raw: Tensor) -> Tensor:
    if output == "TDryBul":
        return raw + 273.15  # From_degC: u - Modelica.Constants.T_zero
    if output == "relHum":
        return 0.01 * raw  # conRelHum, Gain(k = 0.01)
    if output == "winDir":
        return (math.pi / 180.0) * raw  # From_deg
    return raw


def _limit(output: str, value: Tensor) -> Tensor:
    if output == "relHum":
        return torch.clamp(value, 0.0, 1.0)  # min(uMax, max(uMin, u))
    if output == "winSpe":
        return torch.clamp(value, min=0.0)  # max(0, u)
    if output == "winDir":
        return torch.remainder(value, 2 * math.pi)  # mod(u, 2 pi)
    return value


def _check(where: str, out: dict[str, Tensor], times: Tensor) -> None:
    for output, (lo, hi), unit, block in (
        ("TDryBul", T_BOUNDS, "K", "CheckDryBulbTemperature.mo"),
        ("pAtm", P_BOUNDS, "Pa", "CheckPressure.mo"),
    ):
        if output not in out:
            continue
        v = out[output]
        bad = ~((v > lo) & (v < hi))
        if bool(bad.any()):
            k = int(bad.nonzero()[0])
            raise ModelicaImportError(
                f"{where}: {output} = {float(v[k]):g} {unit} at t = {float(times[k]):g} s is "
                f"outside ({lo:g}, {hi:g}) {unit}, where MBL stops ({block})"
            )


def reader_outputs(comp: Component, doc: ModelicaDoc, times,
                   signal: Callable[[str], Tensor | None],
                   outputs: Sequence[str] = WEATHER_BUS_OUTPUTS) -> dict[str, Tensor]:
    """The weather-bus `outputs` of reader `comp` at model times `times` (s): `TDryBul` (K),
    `pAtm` (Pa), `relHum` (1), `winSpe` (m/s), `winDir` (rad), each shaped like `times`
    (module docstring). `signal(input)` returns the series of the signal driving
    `"<comp>.<input>"` (an `Input` source) or `None`. Raises `ModelicaImportError` naming the
    reader."""
    where = f"modelica: {comp.name} ({comp.cls})"
    times = torch.as_tensor(times, dtype=F64)
    sources = {q: source(comp, q) for q in outputs}
    bad = sorted(f"{q}Sou = {s}" for q, s in sources.items()
                 if s not in ("File", "Parameter", "Input"))
    if bad:
        raise ModelicaImportError(f"{where}: {', '.join(bad)} is not supported (File, "
                                  f"Parameter or Input)")
    table = cal = None
    if "File" in sources.values():
        table = read_table(resolve_file(comp, doc))
        first, end = table.span
        if not _repeats(end - first):
            outside = (times < first) | (times - end >= SOLAR_SHIFT)
            if bool(outside.any()):
                raise ModelicaImportError(
                    f"{where}: the weather file covers {first:g} s to {end:g} s and is not a "
                    f"whole number of years long, so it does not repeat; the run reaches "
                    f"t = {float(times[outside][0]):g} s outside it (ConvertTime.mo:22-32)"
                )
        cal = calendar_time(times, float(doc.experiment.get("StartTime", 0.0)), table.span)
    out: dict[str, Tensor] = {}
    for q in outputs:
        s = sources[q]
        if s == "File":
            value = _convert(q, table.spline(FILE_COLUMN[q])(cal))
        elif s == "Parameter":
            value = torch.full_like(times, float(comp.parameters.get(q, DEFAULT_VALUE[q])))
        else:
            driven = signal(f"{q}_in")
            if driven is None:
                raise ModelicaImportError(f"{where}: {q}Sou = Input, but no supported signal "
                                          f"drives {comp.name}.{q}_in")
            value = torch.as_tensor(driven, dtype=F64).expand(times.shape).clone()
        out[q] = _limit(q, value)
    _check(where, out, times)
    return out


def reader_events(comp: Component, doc: ModelicaDoc, lo: float,
                  hi: float) -> tuple[set[float], set[float]]:
    """Model times in `(lo, hi)` at which reader `comp`'s File outputs are not smooth: `jumps`,
    where a repeating file's calendar time wraps (`PartialConvertTime.mo`, a time event), and
    `breaks`, the table's knots (the Akima spline's second derivative jumps there) and the
    points where a limiter starts or stops cutting (`LIMIT_LEVELS`). Empty when no output
    comes from the file."""
    file_outputs = [q for q in WEATHER_BUS_OUTPUTS if source(comp, q) == "File"]
    if not file_outputs:
        return set(), set()
    table = read_table(resolve_file(comp, doc))
    first, end = table.span
    length = end - first
    repeats = _repeats(length)
    jumps: set[float] = set()
    if repeats:
        k0 = math.floor(float(doc.experiment.get("StartTime", 0.0)) / length) + 1
        k_lo, k_hi = max(k0, math.ceil(lo / length)), max(k0, math.ceil(hi / length))
        periods = [(k - 1) * length for k in range(k_lo, k_hi + 1)]
        jumps = {k * length for k in range(k_lo, k_hi) if lo < k * length < hi}
    else:
        periods = [0.0]
    breaks: set[float] = set()
    for off in periods:
        a, b = lo - off, hi - off
        if repeats:
            a, b = max(a, 0.0), min(b, length)
        knots = table.time[(table.time > a) & (table.time < b)] + off
        breaks.update(float(t) for t in knots)
        for q in file_outputs:
            for level in LIMIT_LEVELS.get(q, ()):
                breaks.update(u + off for u in table.spline(FILE_COLUMN[q]).crossings(level, a, b))
    return jumps, {t for t in breaks if lo < t < hi} - jumps
