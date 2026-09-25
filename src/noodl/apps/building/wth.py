"""CONTAM weather file (.wth) reader: TN 1887r1 section 3.15. Tab- or space-separated.

RECORD LAYOUT (the only place this reader's line-consumption assumptions live; verified
against NIST's own `contamxpy` 0.0.9 sample weather files, 15 Sep 2026, one to a year long):

    line 1      `WeatherFile ContamW 2.0` -- the file signature; anything else is refused.
    line 2      a free-text description (ignored).
    line 3      `<start date>\t<end date>` (`M/D` each). Only the start date is used, to
                turn every later `M/D` into a day-of-year offset from it.
    day header  ONE `!Date DofW Dtype DST Tgrnd` comment line, followed by one data row
                PER CALENDAR DAY in the file's date range (`Date DofW Dtype DST Tgrnd`).
                This block is NOT read for its contents -- `DofW`/`Dtype`/`DST`/`Tgrnd`
                play no part in `Ta`/`Pb`/`Ws`/`Wd` -- so no count or per-day line ever
                needs to be checked; it is simply skipped in full.
    time header ONE `!Date Time Ta Pb Ws Wd Hr Ith Idn Ts Rn Sn` comment line, followed by
                ALL the file's per-time data rows for EVERY day, to end of file. This is
                the section actually read: `Date Time Ta Pb Ws Wd` (the rest ignored).

    Verified structural fact, checked directly rather than assumed (the two counted
    sections Task 12 found in the .prj format -- one all on a single line, one whose
    header count was values rather than rows -- have no analogue here: this format
    carries no explicit record count anywhere, so it cannot be miscounted the same way).
    Both header lines occur EXACTLY ONCE per file, regardless of how many days the file
    spans -- checked against every sample in `contamxpy`'s demo set, from one day
    (`valThreeZonesWthCtm.wth`, the committed fixture) to a full year
    (`valThreeZonesWthCtmYear.wth`, 365 day rows, still one time header). A reader that
    consumed one day header per day (as Task 12's contaminants-section bug consumed one
    line per index) would eat the time header itself on day 2 and misparse everything
    after it; this reader instead looks for the marker line itself
    (`startswith("!Date") and "Time" in line`) on every line, not just once, so it is
    unaffected either way a file happens to be laid out.

    CONTAM does not list every time of every day -- only the times at which a value
    changes -- and interpolates linearly between the times it does list (this is also
    what makes the day boundary in a multi-day file carry the SAME instant twice: a
    day's own `24:00:00` row and the next day's `00:00:00` row are both the midnight
    between them, and CONTAM's own sample files write both).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_DAYS_BEFORE_MONTH = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def _day_of_year(date: str) -> int:
    m, d = (int(v) for v in date.split("/"))
    return _DAYS_BEFORE_MONTH[m - 1] + d


def _seconds(hms: str) -> float:
    h, m, s = (int(v) for v in hms.split(":"))
    return 3600.0 * h + 60.0 * m + s


@dataclass
class Weather:
    t: torch.Tensor      # seconds from 00:00 of the file's start date
    Ta: torch.Tensor     # K
    Pb: torch.Tensor     # Pa
    Ws: torch.Tensor     # m/s
    Wd: torch.Tensor     # deg

    def at(self, t: float) -> dict[str, float]:
        """Linear interpolation between listed times (CONTAM interpolates the same way);
        wind direction is interpolated on its shortest arc."""
        tt = self.t.numpy()
        out = {
            name: float(np.interp(t, tt, getattr(self, name).numpy()))
            for name in ("Ta", "Pb", "Ws")
        }
        wd = np.unwrap(np.deg2rad(self.Wd.numpy()))
        out["Wd"] = float(np.rad2deg(np.interp(t, tt, wd)) % 360.0)
        return out

    def drivers_at(self, t: float, *, thermal: str = "thermal") -> dict[str, torch.Tensor]:
        """The weather at `t` under the CONVENTION KEYS a `build_model`-built model reads.

        This is a seam, so the keys are the consumers' own, not this reader's column names:

        * the ambient TEMPERATURE is the thermal layer's prescribed boundary value, so it is
          emitted as `"<thermal>.x_boundary"` -- one entry for the single boundary node
          `thermal_layer` prescribes, hence shape `(1,)`, the shape that layer's `step`
          expects -- and never as a free-standing `"T_amb"`, which nothing in the system
          reads. `thermal` renames it for a layer built through `thermal_layer(name=...)`.
        * the barometric pressure is `"P_ref"`, the key `IdealGasDensity` looks for (CONTAM
          TN 1887r1 section 3.18). It is NOT namespaced: it is a property of the ambient
          state that any closure may read, not of one layer.
        * `"V_met"`/`"theta_w"` are the `Wind` drive's own default `speed_key`/
          `direction_key` and are unchanged.

        A model with more than one prescribed temperature node (`thermal_layer(
        fixed_temperature=...)`) needs more than the ambient value, so it must build the
        boundary vector itself; this returns the one-boundary-node case.
        """
        a = self.at(t)
        f = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
        return {
            f"{thermal}.x_boundary": torch.tensor([a["Ta"]], dtype=torch.float64),
            "P_ref": f(a["Pb"]),
            "V_met": f(a["Ws"]), "theta_w": f(a["Wd"]),
        }


def read_wth(path) -> Weather:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    if not lines or not lines[0].startswith("WeatherFile ContamW"):
        raise ValueError(
            f"wth: {path} is not a CONTAM weather file (expected 'WeatherFile ContamW 2.0')"
        )
    start_date = lines[2].split()[0]
    day0 = _day_of_year(start_date)
    rows = []
    in_data = False
    for ln in lines[3:]:
        if ln.startswith("!Date") and "Time" in ln:
            in_data = True
            continue
        if not in_data or not ln or ln.startswith("!"):
            continue
        tok = ln.split()
        if len(tok) < 6:
            continue
        t = (_day_of_year(tok[0]) - day0) * 86400.0 + _seconds(tok[1])
        # This reader has no year field -- only a day-of-year offset from the file's own
        # start date -- so it cannot tell a window that wraps past 31 December from a file
        # whose dates are simply out of order. A DECREASE here means either: `write_wth`
        # wrote a window that crossed the year boundary (`write_wth` itself now refuses to,
        # but a `.wth` file from elsewhere is not guaranteed to), or the file's rows are not
        # in chronological order. Either way, `Weather.at()`'s `np.interp` needs a monotonic
        # axis, so raise rather than hand it a decreasing one silently.
        if rows and t < rows[-1][0]:
            raise ValueError(
                f"wth: {path} time axis decreases at {tok[0]} {tok[1]} (day-of-year offset "
                f"{t}s follows {rows[-1][0]}s) -- a window crossing 31 December, or rows out "
                f"of order? this reader has no year field to disambiguate"
            )
        rows.append((t, float(tok[2]), float(tok[3]), float(tok[4]), float(tok[5])))
    if not rows:
        raise ValueError(f"wth: {path} has no weather rows")
    cols = list(zip(*rows, strict=True))
    f = lambda c: torch.tensor(c, dtype=torch.float64)  # noqa: E731
    return Weather(t=f(cols[0]), Ta=f(cols[1]), Pb=f(cols[2]), Ws=f(cols[3]), Wd=f(cols[4]))
