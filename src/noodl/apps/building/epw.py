"""EnergyPlus weather (EPW) as the building application's `Weather`, and the CONTAM `.wth`
writer that lets one long sequence drive both the street and the building.

EPW rows are hourly; row (month, day, hour) holds the hour ENDING at `hour` local standard
time, so hour 1 of 1 January is t = 3600 s from 00:00. Only dry-bulb temperature (column 6,
degC), station pressure (9, Pa), wind direction (20, degrees FROM north, clockwise -- the
CONTAM convention) and wind speed (21, m/s) are read.

Day-of-year is computed from `wth.py`'s FIXED, non-leap `_DAYS_BEFORE_MONTH` table --
NEVER from each row's own `year` column via `datetime`/`tm_yday`. A "typical year" (TMY/
IWEC/TMYx) EPW file stitches each month in from a different real source year (a typical
Paris-Orly TMYx file, for instance, has January from 2000, February from 1982, ..., December
from 1977), and those source years' leap status differs from month to month. Computing
day-of-year with the real calendar (which inserts a 29 February whenever that row's OWN year
happens to be a leap year) makes the day count jump by +-1 day at month boundaries where the
leap status changes on either side, breaking the monotonic `t` that `Weather.at()`'s
`np.interp` requires. The fixed table sidesteps this for a genuine "typical year" file, whose
twelve months are stitched in CALENDAR order (January once, February once, ..., December
once): a 29 February row therefore has nowhere to go under this fixed table (real TMY files
do not carry one; a stitched leap-year February is truncated to 28 days in practice) and is
dropped.

The fixed table does NOT, by itself, protect against a file that is not a single stitched
typical year -- a multi-year AMY file, or two TMY files concatenated, repeats a month more
than once, so the fixed day-of-year sequence repeats or goes backwards even though every row
is individually well-formed. `read_epw` checks the result rather than trusting the input: it
raises `ValueError` if the assembled `t` is not strictly increasing, so a genuinely non-
monotonic input fails loudly instead of handing `Weather.at()` a broken interpolation axis
silently (final whole-branch review, finding 6). It also rejects EPW's own missing-value
sentinels on the four columns read here (dry-bulb temperature 99.9 degC, station pressure
999999 Pa, wind direction 999 deg, wind speed 999.0 m/s -- EPW data dictionary, `energyplus.
net`/`bigladdersoftware.com`): a complete typical-year file carries none of these, and a file
that does is missing data this reader has no policy for silently inventing, so it raises
too.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import torch

from noodl.apps.building.wth import _DAYS_BEFORE_MONTH, Weather

_HEADER_LINES = 8
_COL_TEMP, _COL_PRESSURE, _COL_WD, _COL_WS = 6, 9, 20, 21

# EPW missing-value sentinels for the four columns this reader uses (EPW data dictionary).
# A complete typical-year file, the only kind this reader is documented to support, carries
# none of these; encountering one means either a genuinely incomplete data source or a
# column misread, and this reader has no policy for silently standing in a value, so it
# raises rather than passing a sentinel through as if it were a real reading.
_MISSING_TEMP, _MISSING_PRESSURE, _MISSING_WD, _MISSING_WS = 99.9, 999999.0, 999.0, 999.0


def _fixed_day_of_year(month: int, day: int) -> int:
    """Day of year under `wth.py`'s fixed, non-leap calendar (see the module docstring)."""
    return _DAYS_BEFORE_MONTH[month - 1] + day


# A non-leap year, used only as an arithmetic anchor for turning a "seconds from the start
# date" offset into an M/D string. `wth.read_wth`'s day-of-year table (`_DAYS_BEFORE_MONTH`)
# assumes the standard, non-leap month lengths (31, 28, 31, ...) with no year of its own;
# anchoring at a real leap year (e.g. 2024, which the .epw fixture rows carry in their date
# column) would insert a 29 February that table does not know about and shift every later
# month/day pair by one day relative to what `read_wth` will decode. 2023 has no such day.
_ANCHOR_YEAR = 2023


def read_epw(path, *, start_day: int = 1, n_hours: int | None = None) -> Weather:
    rows = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[_HEADER_LINES:]
    t, Ta, Pb, Ws, Wd = [], [], [], [], []
    for line_no, line in enumerate(rows, start=_HEADER_LINES + 1):
        if not line.strip():
            continue
        c = line.split(",")
        month, day, hour = int(c[1]), int(c[2]), int(c[3])
        if month == 2 and day == 29:
            # A source year stitched into February happened to be a leap year; the fixed
            # calendar this reader uses has no 29 February (see the module docstring), and
            # real TMY/IWEC/TMYx files do not carry one in practice, so it is dropped.
            continue
        doy = _fixed_day_of_year(month, day)
        if doy < start_day:
            continue
        temp_c, pressure = float(c[_COL_TEMP]), float(c[_COL_PRESSURE])
        wd_raw, ws = float(c[_COL_WD]), float(c[_COL_WS])
        if temp_c == _MISSING_TEMP:
            raise ValueError(
                f"read_epw: {path}:{line_no} dry-bulb temperature is the EPW missing-value "
                f"sentinel ({_MISSING_TEMP} degC); this reader has no policy for a file with "
                f"missing data"
            )
        if pressure == _MISSING_PRESSURE:
            raise ValueError(
                f"read_epw: {path}:{line_no} station pressure is the EPW missing-value "
                f"sentinel ({_MISSING_PRESSURE:g} Pa); this reader has no policy for a file "
                f"with missing data"
            )
        if wd_raw == _MISSING_WD:
            raise ValueError(
                f"read_epw: {path}:{line_no} wind direction is the EPW missing-value "
                f"sentinel ({_MISSING_WD:g} deg); this reader has no policy for a file with "
                f"missing data"
            )
        if ws == _MISSING_WS:
            raise ValueError(
                f"read_epw: {path}:{line_no} wind speed is the EPW missing-value sentinel "
                f"({_MISSING_WS:g} m/s); this reader has no policy for a file with missing "
                f"data"
            )
        seconds = (doy - start_day) * 86400.0 + hour * 3600.0
        if t and seconds <= t[-1]:
            raise ValueError(
                f"read_epw: {path}:{line_no} time axis is not strictly increasing under the "
                f"fixed non-leap calendar ({seconds}s follows {t[-1]}s) -- more than one "
                f"year of rows? (a multi-year AMY file, or two TMY/IWEC/TMYx files "
                f"concatenated, repeats a calendar month, which this reader's fixed "
                f"day-of-year table cannot represent as a single monotonic axis)"
            )
        t.append(seconds)
        Ta.append(temp_c + 273.15)
        Pb.append(pressure)
        Wd.append(wd_raw % 360.0)
        Ws.append(ws)
        if n_hours is not None and len(t) >= n_hours:
            break
    if not t:
        raise ValueError(f"read_epw: no rows at or after day {start_day} in {path}")
    f = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    return Weather(t=f(t), Ta=f(Ta), Pb=f(Pb), Ws=f(Ws), Wd=f(Wd))


def write_wth(weather: Weather, path, *, start_date: str = "1/1") -> Path:
    """`WeatherFile ContamW 2.0`, one row per sample, the columns `read_wth` reads.

    Header layout (must match `wth.read_wth`'s record layout exactly):
    line 1 the file signature, line 2 free-text description (ignored), line 3
    `"<start date>\\t<end date>"` (only the start date is used, to turn every later `M/D`
    into a day-of-year offset from it), then ONE day-header comment line (`read_wth` skips
    everything up to its own time-header marker, so no per-day rows are needed here) and
    ONE time-header comment line, followed by the data rows: `Date Time Ta Pb Ws Wd` plus
    zeros for the humidity, solar and ground columns `read_wth` ignores.

    A window that crosses 31 December cannot be written: `_ANCHOR_YEAR` is a fixed,
    year-less anchor (see its own docstring), and if `start_date` plus `weather`'s duration
    rolls into the next real calendar year, the `M/D` pairs this writes would repeat a date
    already used earlier in the file (e.g. a run from 12/20 for 20 days writes 1/1..1/9
    twice: once for the real 1 January and again where day 366 wraps), which `read_wth`
    cannot tell apart -- it has no year field, only day-of-year from the file's own start
    date. Raise rather than write a file `read_wth` would silently misread.
    """
    path = Path(path)
    month, day = (int(s) for s in start_date.split("/"))
    base = _dt.datetime(_ANCHOR_YEAR, month, day)
    n = weather.t.numel()
    end_stamp = base + _dt.timedelta(seconds=float(weather.t[-1])) if n else base
    if end_stamp.year != base.year:
        raise ValueError(
            f"write_wth: the window from {start_date} for {float(weather.t[-1]) / 86400.0:.2f}"
            f" days extends past 31 December (to {end_stamp.month}/{end_stamp.day}); wth's "
            f"fixed, year-less day-of-year calendar cannot represent a year wraparound (see "
            f"this function's own docstring)"
        )
    lines = [
        "WeatherFile ContamW 2.0",
        "Generated by noodl.apps.building.epw.write_wth",
        f"{month}/{day}\t{end_stamp.month}/{end_stamp.day}",
        "!Date\tDofW\tDtype\tDST\tTgrnd",
        "!Date\tTime\tTa [K]\tPb [Pa]\tWs [m/s]\tWd [deg]\tHr [g/kg]\tIts [W/m^2]"
        "\tIdn [W/m^2]\tTs [K]\tRn\tSn",
    ]
    for k in range(n):
        stamp = base + _dt.timedelta(seconds=float(weather.t[k]))
        lines.append("\t".join([
            f"{stamp.month}/{stamp.day}", stamp.strftime("%H:%M:%S"),
            f"{weather.Ta[k].item():.3f}", f"{weather.Pb[k].item():.1f}",
            f"{weather.Ws[k].item():.3f}", f"{weather.Wd[k].item():.1f}",
            "0", "0", "0", "0", "0", "0",
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
