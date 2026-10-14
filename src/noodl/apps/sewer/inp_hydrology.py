"""The hydrology sections of a SWMM `.inp`: `[RAINGAGES]`, `[SUBCATCHMENTS]`, `[SUBAREAS]`,
`[INFILTRATION]`, `[EVAPORATION]` and `[ADJUSTMENTS]`, read into a `runoff.Hydrology`.

READ (SWMM 5.2.4 `gage.c`, `subcatch.c`, `infil.c`, `climate.c`; the field layout and the
keyword matching are SWMM's -- a keyword matches when it is the start of the field, as
``match`` in input.c):

- ``[RAINGAGES]`` ``Name Format Interval SCF TIMESERIES Series`` with Format ``INTENSITY``,
  ``VOLUME`` or ``CUMULATIVE``, Interval in decimal hours or ``H:MM`` and the rain unit that of
  the file's unit system (inches, or millimetres). A ``FILE`` source is refused: the rain
  file formats are not modelled. A gage no subcatchment names is ignored.
- ``[SUBCATCHMENTS]`` ``Name Gage Outlet Area %Imperv Width %Slope CurbLen [Snowpack]``: the
  outlet is a node or another subcatchment; the curb length only feeds street sweeping.
- ``[SUBAREAS]`` ``Name N-Imperv N-Perv S-Imperv S-Perv PctZero RouteTo [PctRouted]``.
- ``[INFILTRATION]`` ``Name p1 p2 ... [Method]`` for ``HORTON``, ``MODIFIED_HORTON``
  (``MaxRate MinRate Decay DryTime [MaxInfil]``), ``GREEN_AMPT``, ``MODIFIED_GREEN_AMPT``
  (``Suction Conductivity InitialDeficit``) and ``CURVE_NUMBER`` (``CN Conductivity DryTime``),
  with the model of `[OPTIONS]` ``INFILTRATION`` (default ``HORTON``) unless the line ends in a
  method. A subcatchment with no line has no infiltration.
- ``[EVAPORATION]``: ``CONSTANT``, ``MONTHLY``, ``TIMESERIES``, ``RECOVERY`` and ``DRY_ONLY``;
  ``TEMPERATURE`` and ``FILE`` (a climate file) are refused.
- ``[ADJUSTMENTS]``: ``EVAPORATION``, ``RAINFALL``, ``CONDUCTIVITY``, ``N-PERV``, ``DSTORE``,
  ``INFIL``; ``TEMPERATURE`` is ignored (nothing reads a temperature).
- the options ``WET_STEP``, ``DRY_STEP``, ``START_DATE``/``START_TIME``,
  ``END_DATE``/``END_TIME`` (SWMM's total duration), ``INFILTRATION`` and ``IGNORE_RAINFALL``,
  ``IGNORE_SNOWMELT``, ``IGNORE_GROUNDWATER``, ``IGNORE_QUALITY``.

REFUSED BY NAME (each names its line): ``[LID_USAGE]``; ``[GROUNDWATER]`` unless
``IGNORE_GROUNDWATER YES``; a subcatchment with a snow pack unless ``IGNORE_SNOWMELT YES``;
``[LANDUSES]``, ``[COVERAGES]``, ``[BUILDUP]``, ``[WASHOFF]``, ``[LOADINGS]`` and a rain
concentration of a pollutant (their pollutant load is not modelled), unless ``IGNORE_QUALITY
YES`` or the file has no ``[POLLUTANTS]``; an outfall that routes onto a subcatchment;
an ``[EVAPORATION]`` read from a climate file or computed from temperature; a gage that reads
a rain ``FILE``. SWMM's own errors are refused too (an unknown name, a negative value, an
infiltration parameter out of range, a rain gage interval longer than its series' spacing).
Quietly inert, as in SWMM: ``[SNOWPACKS]`` no subcatchment names, ``[TEMPERATURE]``,
``[AQUIFERS]`` no ``[GROUNDWATER]`` uses and ``[LID_CONTROLS]`` no ``[LID_USAGE]`` places.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Callable

from noodl.apps.inpfile import InpLine
from noodl.apps.sewer.inflows import (
    TimePattern,
    TimeSeries,
    parse_clock,
    parse_date,
    patterns_from_lines,
    tokens,
)
from noodl.apps.sewer.runoff import (
    UCF_LANDAREA,
    UCF_LENGTH,
    UCF_RAINDEPTH,
    UCF_RAINFALL,
    Adjustments,
    CurveNumber,
    Evaporation,
    GreenAmpt,
    Horton,
    Hydrology,
    RainGage,
    Subareas,
    Subcatchment,
)
from noodl.apps.sewer.xsect_tables import LCF

#: The sections this reader takes over when the file has `[SUBCATCHMENTS]`; any other section
#: with content is still refused by `read_swmm_inp`.
HYDROLOGY_SECTIONS = frozenset({
    "RAINGAGES", "SUBCATCHMENTS", "SUBAREAS", "INFILTRATION", "ADJUSTMENTS", "SNOWPACKS",
    "TEMPERATURE", "AQUIFERS", "GROUNDWATER", "GWF", "LID_CONTROLS", "LID_USAGE", "LANDUSES",
    "COVERAGES", "BUILDUP", "WASHOFF", "LOADINGS", "EVAPORATION",
})
_INFIL_MODELS = ("HORTON", "MODIFIED_HORTON", "GREEN_AMPT", "MODIFIED_GREEN_AMPT",
                 "CURVE_NUMBER")
_ROUTE_WORDS = (("OUTLET", "outlet"), ("IMPERV", "impervious"), ("PERV", "pervious"))
_RAIN_WORDS = (("INTENSITY", "intensity"), ("VOLUME", "volume"), ("CUMULATIVE", "cumulative"))
_TOL_QUALITY = ("LANDUSES", "COVERAGES", "BUILDUP", "WASHOFF", "LOADINGS")


def _match(word: str, keywords) -> str | None:
    """SWMM's ``findmatch``: the value of the first keyword that `word` starts with (not case
    sensitive); keywords are ``str`` or ``(str, value)``."""
    up = word.strip().upper()
    for kw in keywords:
        text, value = (kw, kw) if isinstance(kw, str) else kw
        if up.startswith(text):
            return value
    return None


def _fail(path, line: InpLine, message: str):
    raise ValueError(f"{path}: line {line.number}: {message}")


def _number(path, line: InpLine, toks: list[str], i: int, what: str) -> float:
    try:
        return float(toks[i])
    except (IndexError, ValueError):
        got = toks[i] if i < len(toks) else "<missing>"
        raise ValueError(f"{path}: line {line.number} of [{line.section}]: {what} must be a "
                         f"number, got {got!r}") from None


def _flag(options: dict, key: str, path) -> bool:
    if key not in options:
        return False
    line, word = options[key]
    if word not in ("YES", "NO"):
        _fail(path, line, f"{key} must be YES or NO, got {line.fields[1]!r}")
    return word == "YES"


def _time_option(options: dict, key: str, default: float, path) -> float:
    """`[OPTIONS]` time `key` in seconds, read as SWMM's ``datetime_strToTime`` does: decimal
    hours, or ``H:M[:S]``."""
    if key not in options:
        return default
    line, _ = options[key]
    seconds = parse_clock(line.fields[1])
    if seconds is None or seconds < 0.0:
        _fail(path, line, f"{key} must be a time (H:MM:SS or decimal hours), got "
                          f"{line.fields[1]!r}")
    return seconds


def swmm_total_duration(start: datetime.datetime, end: datetime.datetime) -> float:
    """SWMM's ``TotalDuration`` (s) of a run from `start` to `end`:
    ``floor((EndDateTime - StartDateTime) * 86400)`` (project.c:165) with the dates and times as
    SWMM encodes them (days, and the time of day as ``seconds / 86400``). The SWMM 5.2.4 engine
    that pyswmm ships evaluates the difference as ``(EndDate - StartDate) + (EndTime -
    StartTime)`` (the sum of two exact differences rather than the difference of two sums,
    which the floor would take a second lower in a third of the cases: 12:20:00 after
    12:00:00 is 1199 s -- found by comparing the last runoff step with SWMM's), and so does
    this. The result is not positive when `end` is not after `start`."""
    def of_day(t: datetime.datetime) -> float:
        return (t - datetime.datetime.combine(t.date(), datetime.time())
                ).total_seconds() / 86400.0

    span = float((end.date() - start.date()).days) + (of_day(end) - of_day(start))
    return float(math.floor(span * 86400.0))


def _total_duration(options: dict, start: datetime.datetime, path) -> float:
    """`swmm_total_duration` for the run `[OPTIONS]` describe; refuses an end that is not after
    the start."""
    end_date = datetime.date(2004, 1, 1)
    if "END_DATE" in options:
        end_date = parse_date(options["END_DATE"][0].fields[1])
        if end_date is None:
            _fail(path, options["END_DATE"][0], "END_DATE must be a date")
    end = datetime.datetime.combine(end_date, datetime.time()) + datetime.timedelta(
        seconds=_time_option(options, "END_TIME", 0.0, path))
    total = swmm_total_duration(start, end)
    if total <= 0.0:
        line = options["END_DATE"][0] if "END_DATE" in options else None
        raise ValueError(f"{path}: " + (f"line {line.number}: " if line else "")
                         + "the simulation must end after it starts (END_DATE and END_TIME "
                           "are not after START_DATE and START_TIME; SWMM's ERR_START_DATE)")
    return total


def read_hydrology(sections: dict, path, options: dict, start: datetime.datetime, *,
                   nodes: set[str], outfalls: set[str], us: bool,
                   read_series: Callable[[str], TimeSeries], has_pollutants: bool,
                   pollutant_rain: bool, conduit_check: Callable[[], None] | None = None
                   ) -> Hydrology:
    """The `Hydrology` of `sections` (the file `path`).

    `options` is ``{KEY: (InpLine, UPPERCASE VALUE)}`` of `[OPTIONS]`, `start` the simulation's
    start, `nodes` the names of the junctions, dividers and storage units and `outfalls`
    those of the outfalls (the possible outlets), `us` whether the file uses US units
    (inches, feet) rather than SI, `read_series(name)` the `TimeSeries` of a `[TIMESERIES]`
    name (SWMM's rules, `inflows.py`), `has_pollutants` whether `[POLLUTANTS]` exists and
    `pollutant_rain` whether one has a rain concentration. Refusals are in the module
    docstring."""
    ignore_rain = _flag(options, "IGNORE_RAINFALL", path)
    ignore_snow = _flag(options, "IGNORE_SNOWMELT", path)
    ignore_gw = _flag(options, "IGNORE_GROUNDWATER", path)
    ignore_quality = _flag(options, "IGNORE_QUALITY", path)

    for key, why in (("LID_USAGE", "low-impact development units are not modelled"),):
        if sections.get(key):
            _fail(path, sections[key][0], f"[{key}] {why}")
    if sections.get("GROUNDWATER") and not ignore_gw:
        _fail(path, sections["GROUNDWATER"][0],
              "[GROUNDWATER] is not modelled (groundwater flow and the infiltration limit of "
              "its aquifer); set IGNORE_GROUNDWATER YES to run without it")
    if has_pollutants and not ignore_quality:
        for key in _TOL_QUALITY:
            if sections.get(key):
                _fail(path, sections[key][0],
                      f"[{key}] needs the pollutant buildup and washoff of the runoff, which "
                      f"is not modelled; set IGNORE_QUALITY YES to run without it")
        if pollutant_rain:
            _fail(path, sections["POLLUTANTS"][0] if sections.get("POLLUTANTS") else
                  next(iter(sections["SUBCATCHMENTS"])),
                  "a pollutant with a rain concentration (Crain) needs the pollutant load "
                  "of the rainfall, which is not modelled; set IGNORE_QUALITY YES")
    for line in sections.get("OUTFALLS", []):
        kind = line.fields[2].upper() if len(line.fields) > 2 else "FREE"
        at = 4 if kind in ("FREE", "NORMAL") else 5
        if len(line.fields) > at and line.fields[at] != "*":
            _fail(path, line, f"outfall {line.fields[0]!r} routes its flow onto a "
                              f"subcatchment (Route To), which is not modelled")

    patterns = patterns_from_lines(
        [(line.number, tokens(line.raw)) for line in sections.get("PATTERNS", [])
         if tokens(line.raw)], path)

    def monthly(label: str, line: InpLine) -> tuple[float, ...] | None:
        if label not in patterns:
            _fail(path, line, f"names the pattern {label!r}, which has no [PATTERNS] entry")
        p: TimePattern = patterns[label]
        return p.factors if p.kind == "monthly" else None

    # --- rain gages -----------------------------------------------------------------------
    gage_lines: dict[str, InpLine] = {}
    gages: list[RainGage] = []
    series_used: dict[str, TimeSeries] = {}
    named = {tokens(line.raw)[1] for line in sections.get("SUBCATCHMENTS", [])
             if len(tokens(line.raw)) > 1}
    for line in sections.get("RAINGAGES", []):
        toks = tokens(line.raw)
        if toks and toks[0] not in named:
            continue                        # SWMM validates a gage no subcatchment uses no further
        if len(toks) < 5:
            _fail(path, line, "a rain gage needs a name, format, interval, snow catch factor "
                              "and data source")
        name = toks[0]
        source = _match(toks[4], ("TIMESERIES", "FILE"))
        if source is None:
            _fail(path, line, f"rain gage {name!r}: data source must be TIMESERIES or FILE, "
                              f"got {toks[4]!r}")
        if source == "FILE":
            _fail(path, line, f"rain gage {name!r} reads its rainfall from a file, which is "
                              f"not modelled (only TIMESERIES gages are)")
        if len(toks) < 6:
            _fail(path, line, f"rain gage {name!r}: TIMESERIES needs a series name")
        kind = _match(toks[1], _RAIN_WORDS)
        if kind is None:
            _fail(path, line, f"rain gage {name!r}: format must be INTENSITY, VOLUME or "
                              f"CUMULATIVE, got {toks[1]!r}")
        interval = parse_clock(toks[2])
        if interval is None or math.floor(interval + 0.5) <= 0:
            _fail(path, line, f"rain gage {name!r}: bad recording interval {toks[2]!r}")
        snow = _number(path, line, toks, 3, "the snow catch factor")
        if toks[5] not in series_used:
            try:
                series_used[toks[5]] = read_series(toks[5])
            except KeyError:
                _fail(path, line, f"rain gage {name!r} names the time series {toks[5]!r}, "
                                  f"which has no [TIMESERIES] entry")
        if name in gage_lines:
            _fail(path, line, f"rain gage {name!r} is defined twice")
        gage_lines[name] = line
        gages.append(RainGage(name, series_used[toks[5]], kind,
                              float(math.floor(interval + 0.5)), snow))

    # --- evaporation (series before the gages that might share them are checked) -----------
    evaporation, evap_series = _evaporation(sections.get("EVAPORATION", []), path,
                                            read_series, monthly)
    for g in gages:
        if g.series.name in evap_series:
            _fail(path, gage_lines[g.name], f"rain gage {g.name!r} uses the time series "
                                            f"{g.series.name!r}, which gives the evaporation")

    # --- subcatchments ----------------------------------------------------------------------
    sub_lines: dict[str, InpLine] = {}
    raw_subs = []
    for line in sections.get("SUBCATCHMENTS", []):
        toks = tokens(line.raw)
        if len(toks) < 8:
            _fail(path, line, "a subcatchment needs a name, rain gage, outlet, area, "
                              "%imperv, width, %slope and curb length")
        name = toks[0]
        if name in sub_lines:
            _fail(path, line, f"subcatchment {name!r} is defined twice")
        sub_lines[name] = line
        values = []
        for i in range(3, 8):
            v = _number(path, line, toks, i, "a subcatchment parameter")
            if v < 0.0:
                _fail(path, line, f"subcatchment {name!r}: {toks[i]!r} is negative")
            values.append(v)
        if len(toks) > 8 and not ignore_snow:
            _fail(path, line, f"subcatchment {name!r} has the snow pack {toks[8]!r}, and "
                              f"snowmelt is not modelled; set IGNORE_SNOWMELT YES to run "
                              f"without it")
        raw_subs.append((line, toks, values))
    sub_names = set(sub_lines)
    gage_names = {g.name for g in gages}

    areas = {}                                  # subareas
    for line in sections.get("SUBAREAS", []):
        toks = tokens(line.raw)
        if len(toks) < 7:
            _fail(path, line, "a subarea needs a subcatchment, N-Imperv, N-Perv, S-Imperv, "
                              "S-Perv, PctZero and RouteTo")
        if toks[0] not in sub_names:
            _fail(path, line, f"[SUBAREAS] names {toks[0]!r}, which is not a subcatchment")
        if line.number < sub_lines[toks[0]].number:
            _fail(path, line, f"[SUBAREAS] gives subcatchment {toks[0]!r} before its line in "
                              f"[SUBCATCHMENTS]; SWMM reads the file in order and needs the "
                              f"percent imperviousness first")
        x = []
        for i in range(1, 6):
            v = _number(path, line, toks, i, "a subarea parameter")
            if v < 0.0:
                _fail(path, line, f"{toks[i]!r} must not be negative")
            x.append(v)
        route = _match(toks[6], _ROUTE_WORDS)
        if route is None:
            _fail(path, line, f"RouteTo must be OUTLET, IMPERVIOUS or PERVIOUS, got "
                              f"{toks[6]!r}")
        routed = 100.0
        if len(toks) >= 8:
            routed = _number(path, line, toks, 7, "PctRouted")
            if not 0.0 <= routed <= 100.0:
                _fail(path, line, f"PctRouted must be between 0 and 100, got {toks[7]!r}")
        depth = UCF_RAINDEPTH[0 if us else 1]
        areas[toks[0]] = Subareas(
            n_imperv=x[0], n_perv=x[1], s_imperv=x[2] / depth * LCF, s_perv=x[3] / depth * LCF,
            pct_zero=x[4], route_to=route, pct_routed=routed)

    default_model = options["INFILTRATION"][1] if "INFILTRATION" in options else "HORTON"
    if _match(default_model, _INFIL_MODELS) is None:
        _fail(path, options["INFILTRATION"][0], f"INFILTRATION must be one of "
                                                f"{', '.join(_INFIL_MODELS)}")
    default_model = _match(default_model, _INFIL_MODELS)
    infil = {}
    for line in sections.get("INFILTRATION", []):
        toks = tokens(line.raw)
        if toks and toks[0] not in sub_names:
            _fail(path, line, f"[INFILTRATION] names {toks[0]!r}, which is not a subcatchment")
        infil[toks[0]] = _infiltration(path, line, toks, default_model, us)

    adjustments, patterned = _adjustments(sections.get("ADJUSTMENTS", []), path, monthly,
                                          sub_lines)

    nodes_all = nodes | outfalls
    subs = []
    for line, toks, v in raw_subs:
        name, gage, outlet = toks[0], toks[1], toks[2]
        if gage not in gage_names:
            _fail(path, line, f"subcatchment {name!r} names the rain gage {gage!r}, which "
                              f"has no [RAINGAGES] entry")
        is_node, is_sub = outlet in nodes_all, outlet in sub_names
        if is_node and is_sub:
            _fail(path, line, f"the outlet {outlet!r} of subcatchment {name!r} is both a "
                              f"node and a subcatchment (SWMM's ERR_SUBCATCH_OUTLET)")
        if not (is_node or is_sub):
            _fail(path, line, f"subcatchment {name!r} has the outlet {outlet!r}, which is "
                              f"neither a node nor a subcatchment")
        area_ft2 = v[0] / UCF_LANDAREA[0 if us else 1]
        width_ft = v[2] / UCF_LENGTH[0 if us else 1]
        pats = patterned.get(name, {})
        subs.append(Subcatchment(
            name=name, gage=gage, outlet=outlet, area=area_ft2 * LCF * LCF,
            pct_imperv=min(v[1], 100.0), width=width_ft * LCF, slope=v[3] / 100.0,
            subareas=areas.get(name), infiltration=infil.get(name),
            outlet_is_subcatchment=is_sub,
            n_perv_pattern=pats.get("N-PERV"), storage_pattern=pats.get("DSTORE"),
            infiltration_pattern=pats.get("INFIL")))

    # --- gage_validate: used gages, shared series, the recording interval and WET_STEP ----
    wet = _time_option(options, "WET_STEP", 300.0, path)
    dry = _time_option(options, "DRY_STEP", 3600.0, path)
    if wet <= 0.0 or dry <= 0.0:
        bad = "WET_STEP" if wet <= 0.0 else "DRY_STEP"
        _fail(path, options[bad][0], f"{bad} must be greater than 0")
    wet = float(math.floor(wet + 0.5))
    dry = float(math.floor(dry + 0.5))
    used = [g for g in gages if any(s.gage == g.name for s in subs)]
    if sections.get("SNOWPACKS") and not ignore_snow:
        # with snow packs defined SWMM turns precipitation below the snow temperature into
        # snowfall scaled by the gage's catch factor (gage_getPrecip); the factor 1 changes nothing
        for g in used:
            if g.snow_factor != 1.0:
                _fail(path, gage_lines[g.name],
                      f"rain gage {g.name!r} has the snow catch factor {g.snow_factor:g}, which "
                      f"scales snowfall; the file has snow packs and snowmelt is not modelled "
                      f"(use 1, or IGNORE_SNOWMELT YES)")
    seen: list[RainGage] = []
    for g in used:
        co = next((o for o in seen if o.series == g.series), None)
        seen.append(g)
        if co is not None:
            if co.kind != g.kind:
                _fail(path, gage_lines[g.name], f"rain gages {co.name!r} and {g.name!r} share "
                                                f"a time series but not a rain format")
            continue
        times = g.series.times
        if len(times) > 1:
            spacing = min(b - a for a, b in zip(times, times[1:], strict=False))
            if g.interval > math.floor(spacing + 0.5) > 0:
                _fail(path, gage_lines[g.name],
                      f"rain gage {g.name!r}: its recording interval is longer than the "
                      f"spacing of its time series (SWMM's ERR_RAIN_GAGE_INTERVAL)")
        if g.interval < wet:
            wet = g.interval
    dry = max(dry, wet)
    total = _total_duration(options, start, path)
    return Hydrology(
        gages=tuple(gages), subcatchments=tuple(subs), start=start, total_duration=total,
        wet_step=wet, dry_step=dry, us_units=us, evaporation=evaporation,
        adjustments=adjustments, ignore_rainfall=ignore_rain)


def read_climate(sections: dict, path, options: dict, start: datetime.datetime, *,
                 us: bool, read_series: Callable[[str], TimeSeries], evaporation: bool = True
                 ) -> Hydrology:
    """The climate of a file with no `[SUBCATCHMENTS]` (a `Hydrology` with no gages or
    subcatchments): `[EVAPORATION]` (unless `evaporation` is False) and `[ADJUSTMENTS]`, read
    as `read_hydrology` reads them. SWMM's conduit seepage, open conduits and storage units
    read the evaporation rate and the conductivity adjustment at every routing step
    (`losses.ClimateClock`). `options`, `start`, `us` and `read_series` are those of
    `read_hydrology`. An ``[ADJUSTMENTS]`` line that names a subcatchment is refused: there
    is none."""
    patterns = patterns_from_lines(
        [(line.number, tokens(line.raw)) for line in sections.get("PATTERNS", [])
         if tokens(line.raw)], path)

    def monthly(label: str, line: InpLine) -> tuple[float, ...] | None:
        if label not in patterns:
            _fail(path, line, f"names the pattern {label!r}, which has no [PATTERNS] entry")
        p: TimePattern = patterns[label]
        return p.factors if p.kind == "monthly" else None

    lines = sections.get("EVAPORATION", []) if evaporation else []
    evap, _ = _evaporation(lines, path, read_series, monthly)
    adjustments, _ = _adjustments(sections.get("ADJUSTMENTS", []), path, monthly, {})
    return Hydrology(gages=(), subcatchments=(), start=start,
                     total_duration=_total_duration(options, start, path), us_units=us,
                     evaporation=evap, adjustments=adjustments)


def _infiltration(path, line: InpLine, toks: list[str], default_model: str, us: bool):
    """An `[INFILTRATION]` line (`infil_readParams`, infil.c:130) as Horton, GreenAmpt or
    CurveNumber in SI."""
    model = default_model
    n = len(toks)
    m = _match(toks[-1], _INFIL_MODELS) if n > 1 else None
    if m is not None:
        model = m
        n -= 1
    need = {"HORTON": 5, "MODIFIED_HORTON": 5, "GREEN_AMPT": 4, "MODIFIED_GREEN_AMPT": 4,
            "CURVE_NUMBER": 4}[model]
    if n < need:
        _fail(path, line, f"[INFILTRATION] {model} needs {need - 1} parameters after the "
                          f"subcatchment")
    x = [0.0] * 5
    for i in range(1, need):
        x[i - 1] = _number(path, line, toks, i, "an infiltration parameter")
    if model in ("HORTON", "MODIFIED_HORTON") and n > need:
        x[need - 1] = _number(path, line, toks, need, "the maximum infiltration volume")
    rate = UCF_RAINFALL[0 if us else 1]
    depth = UCF_RAINDEPTH[0 if us else 1]
    bad = f"infiltration parameters out of range for {model} (SWMM's ERR_INFIL_PARAMS)"
    if model in ("HORTON", "MODIFIED_HORTON"):
        if any(v < 0.0 for v in x) or x[0] / rate < x[1] / rate:
            _fail(path, line, bad)
        dry_days = x[3] if x[3] != 0.0 else 1.0e-6
        return Horton(f0=x[0] / rate * LCF, fmin=x[1] / rate * LCF, decay=x[2] / 3600.0,
                      dry_time=dry_days * 86400.0, max_volume=x[4] / depth * LCF,
                      modified=model == "MODIFIED_HORTON")
    if model in ("GREEN_AMPT", "MODIFIED_GREEN_AMPT"):
        if x[0] < 0.0 or x[1] <= 0.0 or x[2] < 0.0 or x[2] > 1.0:
            _fail(path, line, bad)
        return GreenAmpt(suction=x[0] / depth * LCF, conductivity=x[1] / rate * LCF,
                         initial_deficit=x[2], modified=model == "MODIFIED_GREEN_AMPT")
    if not x[2] > 0.0:
        _fail(path, line, bad)
    return CurveNumber(curve_number=x[0], dry_time=x[2] * 86400.0)


def _evaporation(lines, path, read_series, monthly):
    """`(Evaporation, {series names it uses})` from `[EVAPORATION]` (`climate_readEvapParams`)."""
    kind, values, series, dry_only, recovery = "constant", (0.0,) * 12, None, False, None
    used: set[str] = set()
    for line in lines:
        toks = tokens(line.raw)
        if not toks:
            continue
        word = _match(toks[0], ("CONSTANT", "MONTHLY", "TIMESERIES", "TEMPERATURE", "FILE",
                                "RECOVERY", "DRY_ONLY"))
        if word is None:
            _fail(path, line, f"[EVAPORATION] {toks[0]!r} is not CONSTANT, MONTHLY, "
                              f"TIMESERIES, TEMPERATURE, FILE, RECOVERY or DRY_ONLY")
        if word in ("TEMPERATURE", "FILE"):
            _fail(path, line, f"[EVAPORATION] {word} takes the evaporation from a climate "
                              f"file, which is not modelled")
        if word == "DRY_ONLY":
            if len(toks) < 2:
                _fail(path, line, "[EVAPORATION] DRY_ONLY needs YES or NO")
            flag = toks[1].upper()
            if flag not in ("YES", "NO"):
                _fail(path, line, f"[EVAPORATION] DRY_ONLY must be YES or NO, got {toks[1]!r}")
            dry_only = flag == "YES"
            continue
        if len(toks) < 2:
            _fail(path, line, f"[EVAPORATION] {word} needs a value")
        if word == "RECOVERY":
            recovery = monthly(toks[1], line)
            continue
        if word == "CONSTANT":
            kind = "constant"
            values = (_number(path, line, toks, 1, "the evaporation rate"),) * 12
        elif word == "MONTHLY":
            if len(toks) < 13:
                _fail(path, line, "[EVAPORATION] MONTHLY needs 12 rates")
            kind = "monthly"
            values = tuple(_number(path, line, toks, i, "an evaporation rate")
                           for i in range(1, 13))
        else:
            kind = "timeseries"
            try:
                series = read_series(toks[1])
            except KeyError:
                _fail(path, line, f"[EVAPORATION] names the time series {toks[1]!r}, which "
                                  f"has no [TIMESERIES] entry")
            used.add(toks[1])
    return Evaporation(kind, values, series, dry_only, recovery), used


def _adjustments(lines, path, monthly, sub_lines):
    """`(Adjustments, {subcatchment: {key: monthly factors}})` from `[ADJUSTMENTS]`
    (`climate_readAdjustments`). SWMM reads the file in order and `subcatch_readParams` resets
    the three pattern assignments of a subcatchment, so an ``N-PERV``, ``DSTORE`` or ``INFIL``
    line that comes before the subcatchment's line in `[SUBCATCHMENTS]` has no effect, here as
    there."""
    evap, rain, cond = [0.0] * 12, [1.0] * 12, [1.0] * 12
    patterned: dict[str, dict[str, tuple[float, ...]]] = {}
    for line in lines:
        toks = tokens(line.raw)
        if len(toks) <= 1:
            continue
        key = toks[0].upper()
        if key.startswith("TEMP"):
            continue
        if key.startswith("EVAP") or key.startswith("RAIN") or key.startswith("CONDUCT"):
            if len(toks) < 13:
                _fail(path, line, f"[ADJUSTMENTS] {toks[0]} needs 12 monthly values")
            vals = [_number(path, line, toks, i, "a monthly adjustment") for i in range(1, 13)]
            if key.startswith("EVAP"):
                evap = vals
            elif key.startswith("RAIN"):
                rain = vals
            else:
                cond = [v if v > 0.0 else 1.0 for v in vals]
            continue
        word = next((w for w in ("N-PERV", "DSTORE", "INFIL") if key.startswith(w)), None)
        if word is None:
            _fail(path, line, f"[ADJUSTMENTS] {toks[0]!r} is not TEMPERATURE, EVAPORATION, "
                              f"RAINFALL, CONDUCTIVITY, N-PERV, DSTORE or INFIL")
        if len(toks) < 3:
            _fail(path, line, f"[ADJUSTMENTS] {word} needs a subcatchment and a pattern")
        if toks[1] not in sub_lines:
            _fail(path, line, f"[ADJUSTMENTS] names {toks[1]!r}, which is not a subcatchment")
        factors = monthly(toks[2], line)
        if factors is not None and line.number > sub_lines[toks[1]].number:
            patterned.setdefault(toks[1], {})[word] = factors
    return Adjustments(tuple(evap), tuple(rain), tuple(cond)), patterned


__all__ = ["HYDROLOGY_SECTIONS", "read_climate", "read_hydrology", "swmm_total_duration"]
