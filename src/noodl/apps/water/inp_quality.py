"""The water-quality sections of an EPANET 2.2 `.inp` (shared by `inp.read_epanet_inp`):
`[QUALITY] [SOURCES] [REACTIONS] [MIXING]`, `[OPTIONS] QUALITY|DIFFUSIVITY|TOLERANCE` and
`[TIMES] QUALITY TIMESTEP`, into a `WaterQuality` in SI.

Transcribed from EPANET 2.2 (tag `v2.2`): `input3.c` (`optionchoice`, `optionvalue`,
`qualdata`, `sourcedata`, `reactdata`, `mixingdata`) for what each line means, `input2.c`
(`match`, `findmatch`) for how a keyword is recognised (case-insensitively, the token BEGINNING
with the keyword), `input1.c` (`adjustdata`, `convertunits`) for the units. A later line
overrides an earlier one. EPANET silently skips a few lines (a `[MIXING]` entry on a junction
or a reservoir, a `TANK` coefficient on a junction, a `BULK` or `WALL` one on a pump or valve)
and so does this reader, but it RECORDS each in the returned list (`notes["ignored_quality"]`).
What EPANET refuses it refuses by name (errors 201, 209, 212, 213), and an unknown node, link or
pattern is refused as the rest of this reader refuses it (EPANET: 203, 205, or silently
skipped).
"""

from __future__ import annotations

import re

from noodl.apps.inpfile import InpLine, as_float
from noodl.apps.water.inp_times import time_with_unit
from noodl.apps.water.quality_data import (
    MIXING_MODELS,
    SOURCE_KINDS,
    QualitySource,
    WaterQuality,
)
from noodl.apps.water.units import DIFFUSIVITY_M2, M_PER_FT

#: The `[OPTIONS]` keys this module reads (the reader must not also record them as unknown).
#: EPANET's own keyword strings (`text.h`: `w_QUALITY "QUAL"`, `w_DIFFUSIVITY "DIFF"`,
#: `w_TOLERANCE "TOLER"`), which `match` takes as prefixes of the token: `QUAL`, `DIFF 2` and
#: `TOLER 0.5` are the options too.
OPTION_KEYS = ("QUAL", "DIFF", "TOLER")

_SECONDS_PER_DAY = 86400.0
_ATOL = re.compile(r"\s*([+-]?\d+)")


def match(token: str, keyword: str) -> bool:
    """EPANET's `match` (`input2.c`): `token` begins with `keyword`, ignoring case."""
    return token.upper().startswith(keyword)


def is_quality_option(line: InpLine) -> bool:
    """Whether an `[OPTIONS]` line is one of `OPTION_KEYS` (`match` on its first token)."""
    return any(match(line.fields[0], key) for key in OPTION_KEYS)


def is_quality_time(line: InpLine) -> bool:
    """Whether a `[TIMES]` line is the quality step (`timedata`: the first token matches
    `w_QUALITY`, "QUAL")."""
    return match(line.fields[0], "QUAL")


def _atol(text: str) -> int:
    """C `atol`: the leading integer of `text`, 0 if it has none (`atol("10x") == 10`)."""
    found = _ATOL.match(text)
    return int(found.group(1)) if found else 0


def _in_range(first: str, last: str, names) -> list[str]:
    """The `names` inside the range `first`..`last`, as `qualdata` and `reactdata` read it:
    numerically (C `atol` of each name) when `atol` of both bounds is positive, else by
    `strcmp` order (inclusive)."""
    low, high = _atol(first), _atol(last)
    if low > 0 and high > 0:
        return [n for n in names if low <= _atol(n) <= high]
    return [n for n in names if first <= n <= last]


def _fail(path, line: InpLine, message: str) -> ValueError:
    return ValueError(f"{path}: line {line.number} of [{line.section}]: {message}")


def read_quality(
    sections, path, *, us: bool, junctions, reservoirs, tanks, pipes, pumps, valves, patterns
) -> tuple[WaterQuality, list[str]]:
    """`(WaterQuality, ignored)` for the quality data of the parsed `sections` of `path`.

    `us` is a US-units file (feet: the wall coefficients and the diffusivity carry them).
    The node and link lists name what exists; `patterns` the pattern names. `ignored` holds
    one text per line EPANET skips, for `notes["ignored_quality"]`.
    """
    junction_names = [j.name for j in junctions]
    tank_names = [t.name for t in tanks]
    reservoir_names = {r.name for r in reservoirs}
    link_kind = {p.name: "PIPE" for p in pipes}
    link_kind.update({p.name: "PUMP" for p in pumps})
    link_kind.update({v.name: "VALVE" for v in valves})
    stored = [(line.number, line.fields[0]) for s in ("RESERVOIRS", "TANKS")
              for line in sections.get(s, [])]
    node_order = (*junction_names, *(name for _, name in sorted(stored)))
    numbered = [(line.number, line.fields[0]) for s in ("PIPES", "PUMPS", "VALVES")
                for line in sections.get(s, [])]
    link_order = tuple(name for _, name in sorted(numbered))
    nodes = set(node_order)
    ignored: list[str] = []

    def skip(line: InpLine, why: str) -> None:
        ignored.append(f"[{line.section}] {line.raw.strip()} ({why})")

    # ---- [OPTIONS] and [TIMES]: `optionchoice`, `optionvalue`; `[TIMES] QUALITY TIMESTEP`
    # `optionchoice` overrides the name and units field by field: NONE and CHEM change
    # neither, another word sets the name (and the units only if a units token follows), AGE
    # and TRACE set both; the mode is the last line's.
    mode, chemical, units, trace_node = "NONE", "Chemical", "mg/L", None
    tolerance, diffusivity_file, quality_timestep = 0.01, 1.0, None
    for line in sections.get("OPTIONS", []):
        if not is_quality_option(line):
            continue
        key, n = line.fields[0], len(line.fields)
        if n < 2:
            skip(line, "no value")
        elif match(key, "QUAL"):
            word = line.fields[1]
            if match(word, "NONE"):
                mode = "NONE"
            elif match(word, "CHEM"):
                mode = "CHEMICAL"
            elif match(word, "AGE"):
                mode = "AGE"
            elif match(word, "TRACE"):
                mode = "TRACE"
            else:
                # "any other word [units]": a chemical so named (`optionchoice`)
                mode, chemical = "CHEMICAL", word
                if n >= 3:
                    units = line.fields[2]
            if mode == "AGE":
                chemical, units = "Age", "hrs"
            elif mode == "TRACE":
                if n < 3:
                    raise _fail(path, line, "QUALITY TRACE needs a node (EPANET error 201)")
                if line.fields[2] not in nodes:
                    raise _fail(path, line, f"QUALITY TRACE names the node "
                                f"{line.fields[2]!r}, which is not a node of this network "
                                f"(EPANET error 212)")
                trace_node, units = line.fields[2], "%"
                chemical = f"Trace {trace_node}"
        else:
            value = as_float(line, 1, f"the {key.upper()} value", path)
            if value < 0:
                raise _fail(path, line, f"{key.upper()} {value} is negative (EPANET error 213)")
            if match(key, "TOLER"):
                tolerance = value
            else:
                diffusivity_file = value
    for line in sections.get("TIMES", []):
        if is_quality_time(line):
            # `timedata` sets Qstep for any line whose first token matches QUAL
            index = 2 if len(line.fields) > 2 and match(line.fields[1], "TIME") else 1
            step = time_with_unit(line.fields, index, path, line)
            quality_timestep = step if step > 0 else None

    # ---- [QUALITY]: `qualdata`
    initial: dict[str, float] = {}
    for line in sections.get("QUALITY", []):
        n = len(line.fields)
        if n < 2:
            skip(line, "no value")
            continue
        value = as_float(line, 2 if n > 2 else 1, "the initial quality", path)
        if value < 0:
            raise _fail(path, line, f"negative initial quality {value} (EPANET error 209)")
        if n == 2:
            if line.fields[0] not in nodes:
                raise _fail(path, line, f"names node {line.fields[0]!r}, which is not a "
                            f"node of this network")
            initial[line.fields[0]] = value
        else:
            for name in _in_range(line.fields[0], line.fields[1], node_order):
                initial[name] = value

    # ---- [SOURCES]: `sourcedata`
    sources: dict[str, QualitySource] = {}
    for line in sections.get("SOURCES", []):
        n = len(line.fields)
        if n < 2:
            raise _fail(path, line, "a source needs a node and a strength")
        node = line.fields[0]
        if node not in nodes:
            raise _fail(path, line, f"names node {node!r}, which is not a node of this "
                        f"network")
        kind = next((k for k in SOURCE_KINDS if match(line.fields[1], k)), None)
        if kind is not None:
            if n < 3:
                raise _fail(path, line, f"the {kind} source has no strength")
            strength = as_float(line, 2, "the source strength", path)
            pattern = line.fields[3] if n > 3 else None
        else:
            # the legacy form `node strength [pattern]`, a CONCEN source
            try:
                strength = float(line.fields[1])
            except ValueError:
                raise _fail(path, line, f"{line.fields[1]!r} is neither a source type "
                            f"({', '.join(SOURCE_KINDS)}) nor a strength; unknown source "
                            f"type") from None
            kind = "CONCEN"
            pattern = line.fields[2] if n > 2 else None
        if pattern == "*":
            pattern = None
        if pattern is not None and pattern not in patterns:
            raise _fail(path, line, f"names pattern {pattern!r}, which has no [PATTERNS] "
                        f"entry")
        if kind == "MASS":
            strength /= 60.0          # the file's mass/min; EPANET works in mass/s
        sources[node] = QualitySource(node, kind, strength, pattern)

    # ---- [REACTIONS]: `reactdata` (converted below, once every ORDER is known)
    orders = {"BULK": 1.0, "TANK": 1.0, "WALL": 1.0}
    global_bulk = global_wall = limiting = roughness = 0.0
    bulk: dict[str, float] = {}
    wall: dict[str, float] = {}
    tank_bulk: dict[str, float] = {}
    for line in sections.get("REACTIONS", []):
        n = len(line.fields)
        if n < 3:
            skip(line, "fewer than three fields")
            continue
        first, second = line.fields[0], line.fields[1]
        if match(first, "ORDER"):
            value = as_float(line, n - 1, "the reaction order", path)
            which = next((w for w in orders if match(second, w)), None)
            if which is None:
                raise _fail(path, line, f"ORDER {second!r} is not BULK, TANK or WALL "
                            f"(EPANET error 213)")
            if which == "WALL" and value not in (0.0, 1.0):
                raise _fail(path, line, f"ORDER WALL {value:g} must be 0 or 1 (EPANET "
                            f"error 213)")
            orders[which] = value
        elif match(first, "ROUG"):
            roughness = as_float(line, n - 1, "the roughness correlation", path)
        elif match(first, "LIMIT"):
            limiting = as_float(line, n - 1, "the limiting potential", path)
        elif match(first, "GLOB"):
            value = as_float(line, n - 1, "the global coefficient", path)
            if match(second, "BULK"):
                global_bulk = value
            elif match(second, "WALL"):
                global_wall = value
            else:
                raise _fail(path, line, f"GLOBAL {second.upper()} is not a coefficient "
                            f"EPANET reads: only GLOBAL BULK and GLOBAL WALL (error 213)")
        elif match(first, "TANK"):
            value = as_float(line, n - 1, "the tank coefficient", path)
            stores = [*tank_names, *reservoir_names]
            if n == 3:
                if second in junction_names:
                    skip(line, f"{second!r} is a junction, not a tank")
                elif second not in nodes:
                    raise _fail(path, line, f"names node {second!r}, which is not a node "
                                f"of this network")
                elif second in tank_names:
                    tank_bulk[second] = value
                # a reservoir has no volume and takes none: skipped as `reactdata` does
            else:
                for name in _in_range(second, line.fields[2], [
                        x for x in node_order if x in stores]):
                    if name in tank_names:
                        tank_bulk[name] = value
        elif match(first, "BULK") or match(first, "WALL"):
            value = as_float(line, n - 1, "the pipe coefficient", path)
            target = bulk if match(first, "BULK") else wall
            if n == 3:
                if second not in link_kind:
                    raise _fail(path, line, f"names link {second!r}, which is not a link "
                                f"of this network")
                if link_kind[second] != "PIPE":
                    skip(line, f"{second!r} is a {link_kind[second].lower()}, not a pipe")
                else:
                    target[second] = value
            else:
                for name in _in_range(second, line.fields[2], link_order):
                    if link_kind[name] == "PIPE":
                        target[name] = value
        else:
            raise _fail(path, line, f"unknown [REACTIONS] keyword {first.upper()!r} "
                        f"(EPANET error 213)")

    # ---- [MIXING]: `mixingdata`
    mixing: dict[str, tuple[str, float]] = {}
    for line in sections.get("MIXING", []):
        n = len(line.fields)
        if n < 2:
            skip(line, "no model")
            continue
        node = line.fields[0]
        if node not in nodes:
            raise _fail(path, line, f"names node {node!r}, which is not a node of this "
                        f"network")
        if node in junction_names:      # `mixingdata` returns before it reads the model
            skip(line, f"{node!r} is a junction, not a tank")
            continue
        model = next((m for m in MIXING_MODELS if match(line.fields[1], m)), None)
        if model is None:
            raise _fail(path, line, f"unknown mixing model {line.fields[1]!r}; use "
                        f"{', '.join(MIXING_MODELS)} (EPANET error 213)")
        fraction = 1.0
        if model == "2COMP" and n == 3:
            fraction = as_float(line, 2, "the mixing zone fraction", path)
        if fraction == 0.0:
            fraction = 1.0
        if node not in tank_names:
            skip(line, f"{node!r} is a reservoir, not a tank")
        else:
            mixing[node] = (model, fraction)

    # ---- units: `adjustdata`, `convertunits`. Bulk and tank coefficients per day -> per
    # second; wall order 1 is length per day (ft in a US file), order 0 mass per area per day;
    # the roughness correlation is a wall coefficient too.
    length = M_PER_FT if us else 1.0
    if orders["WALL"] == 1.0:
        wall_factor = length / _SECONDS_PER_DAY
    else:
        wall_factor = 1.0 / length**2 / _SECONDS_PER_DAY
    # DIFFUSIVITY above 1e-4 multiplies water's own, else it is the value in ft2/s or m2/s
    if diffusivity_file > 1e-4:
        diffusivity = diffusivity_file * DIFFUSIVITY_M2
    else:
        diffusivity = diffusivity_file * (length**2)

    quality = WaterQuality(
        mode=mode, chemical=chemical, units=units,
        trace_node=trace_node if mode == "TRACE" else None,
        quality_timestep=quality_timestep, tolerance=tolerance, diffusivity=diffusivity,
        bulk_order=orders["BULK"], wall_order=orders["WALL"], tank_order=orders["TANK"],
        global_bulk=global_bulk / _SECONDS_PER_DAY, global_wall=global_wall * wall_factor,
        pipe_bulk={k: v / _SECONDS_PER_DAY for k, v in bulk.items()},
        pipe_wall={k: v * wall_factor for k, v in wall.items()},
        tank_bulk={k: v / _SECONDS_PER_DAY for k, v in tank_bulk.items()},
        limiting_potential=limiting, roughness_correlation=roughness * wall_factor,
        initial=initial, sources=tuple(sources.values()), mixing=mixing,
        node_order=tuple(node_order), link_order=link_order,
    )
    return quality, ignored
