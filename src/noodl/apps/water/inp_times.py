"""Time and clock-time fields of an EPANET `.inp` (shared by the water reader and its
rule parser): `HH:MM[:SS]`, a bare number of hours, an optional unit token, `AM`/`PM`."""

from __future__ import annotations


def time_of(text: str, path, line) -> float:
    """`HH:MM`, `HH:MM:SS` or a bare number of HOURS -> seconds (Appendix C.2.24)."""
    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(
                f"{path}: line {line.number}: {text!r} is not a time of the form "
                f"HH:MM[:SS]"
            )
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                f"{path}: line {line.number}: {text!r} is not a time of the form "
                f"HH:MM[:SS]"
            ) from exc
        while len(values) < 3:
            values.append(0.0)
        return values[0] * 3600.0 + values[1] * 60.0 + values[2]
    try:
        return float(text) * 3600.0
    except ValueError as exc:
        raise ValueError(
            f"{path}: line {line.number}: {text!r} is not a duration"
        ) from exc


# EPANET (input3.c `hour`, `match`) takes a unit token whose FIRST characters are SEC, MIN, HOU
# or DAY, case-insensitively: MINS, SECS, HOUR, HOURS, HOU and DAYSX all match.
_TIME_UNITS = (("SEC", 1.0), ("MIN", 60.0), ("HOU", 3600.0), ("DAY", 86400.0))


def time_unit_scale(token: str) -> float | None:
    """Seconds per unit for `token`, by EPANET's leading-characters rule, else None."""
    upper = token.upper()
    for prefix, scale in _TIME_UNITS:
        if upper.startswith(prefix):
            return scale
    return None


def time_with_unit(fields, index: int, path, line) -> float:
    """A `[TIMES]` value `fields[index]` with its optional unit token `fields[index + 1]`.

    A bare number without a unit is HOURS (Appendix C.2.24); a unit token starting SEC, MIN,
    HOU or DAY (EPANET's rule, any case: MINS, SECS, HOU) scales it. A clock time
    (`HH:MM[:SS]`) carries no unit. Any other trailing
    token is refused rather than read as a different duration.
    """
    if index >= len(fields):
        raise ValueError(f"{path}: line {line.number}: [TIMES] entry has no value")
    value, extra = fields[index], list(fields[index + 1:])
    if not extra:
        return time_of(value, path, line)
    if len(extra) > 1 or ":" in value:
        raise ValueError(
            f"{path}: line {line.number}: cannot read {' '.join(fields[index:])!r} as a "
            f"time: expected HH:MM[:SS] or a number with an optional unit"
        )
    scale = time_unit_scale(extra[0])
    if scale is None:
        raise ValueError(
            f"{path}: line {line.number}: unknown time unit {extra[0]!r}; use SECONDS, "
            f"MINUTES, HOURS or DAYS"
        )
    try:
        return float(value) * scale
    except ValueError as exc:
        raise ValueError(
            f"{path}: line {line.number}: {value!r} is not a duration"
        ) from exc


def clock_time(text: str, meridiem: str | None, path, line) -> float:
    """A clock time of day, `H[:MM[:SS]]` with an optional `AM`/`PM`, in seconds after
    midnight. `12 AM` is 0 and `12 PM` is noon; without a meridiem the time is on the 24-hour
    clock (`18` is 6 PM), as EPANET reads it (`input3.c`, `gettime` and `hour`)."""
    seconds = time_of(text, path, line)
    if meridiem is None:
        return seconds % 86400.0
    upper = meridiem.upper()
    if upper not in ("AM", "PM") or seconds >= 13 * 3600.0:
        raise ValueError(
            f"{path}: line {line.number}: {text} {meridiem!r} is not a clock time; use "
            f"H[:MM[:SS]] with AM or PM, or the 24-hour clock without"
        )
    if upper == "AM":
        return seconds - 12 * 3600.0 if seconds >= 12 * 3600.0 else seconds
    return seconds + 12 * 3600.0 if seconds < 12 * 3600.0 else seconds
