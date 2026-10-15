"""The RDII sections of a SWMM `.inp`: `[HYDROGRAPHS]` and `[RDII]`, read into an `rdii.Rdii`
(SWMM 5.2.4 ``rdii_readUnitHydParams`` and ``rdii_readRdiiInflow``, ``rdii.c``).

READ:

- ``[HYDROGRAPHS]`` ``Name Gage`` gives the unit hydrograph set `Name` its rain gage (any
  ``[RAINGAGES]`` name; SWMM marks that gage used, so it also lowers WET_STEP and feeds the
  runoff steps); ``Name Month UnitHydrograph R T K [Dmax Drecov Dinit]`` gives the triangle of
  `UnitHydrograph` (SHORT, MEDIUM or LONG; the word must START with it) in `Month` (JAN to DEC, the
  first three letters, or ``ALL``): the response ratio R, the time to peak T and the recession
  ratio K (hours; the base is ``T * (1 + K)``) and the optional initial abstraction (the rain
  depth units of the file; Drecov per day). The older ``Name Month R1 T1 K1 R2 T2 K2 R3 T3 K3
  [Dmax Drecov Dinit]`` line is read too. Lines apply in order: ``ALL`` then a month replaces
  that month only.
- ``[RDII]`` ``Node UnitHydrograph SewerArea`` (acres, or hectares in SI units, as SWMM's
  land-area unit); a later line for the same node replaces the earlier.
- ``IGNORE_RDII YES`` (no RDII is computed, the gages stay used).

Names are not case sensitive, as in SWMM (hash.c).

REFUSED BY NAME: a set a node uses (or any set, when a node has RDII) without a gage line
(SWMM would read the first gage of the file), a gage that is not in ``[RAINGAGES]``, negative
ratios or times, ratios that add to more than 1.01 in a month, a negative area, an unknown
node or set, and (in ``read_swmm_inp``) ``USE RDII`` in ``[FILES]``.
"""

from __future__ import annotations

from noodl.apps.inpfile import InpLine
from noodl.apps.sewer.inflows import tokens
from noodl.apps.sewer.rdii import (
    MONTH_NAMES,
    RESPONSES,
    Rdii,
    RdiiInflow,
    UnitHydrograph,
    UnitHydrographSet,
)
from noodl.apps.sewer.runoff import UCF_LANDAREA
from noodl.apps.sewer.xsect_tables import LCF


def _fail(path, line: InpLine, message: str):
    raise ValueError(f"{path}: line {line.number}: {message}")


def _num(path, line: InpLine, text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        _fail(path, line, f"{what} must be a number, got {text!r}")


def read_rdii(sections: dict, path, *, nodes: set[str], gages: set[str], us: bool,
              ignore: bool) -> Rdii | None:
    """The `Rdii` of `[HYDROGRAPHS]` and `[RDII]` in `sections` (the file `path`), or ``None``
    when the file has neither. `nodes` are the names of all nodes, `gages` those of all
    ``[RAINGAGES]``, `us` whether the file's area unit is acres (else hectares) and `ignore`
    IGNORE_RDII."""
    hyd_lines = [(line, tokens(line.raw)) for line in sections.get("HYDROGRAPHS", [])]
    hyd_lines = [(line, toks) for line, toks in hyd_lines if toks]
    rdii_lines = [(line, tokens(line.raw)) for line in sections.get("RDII", [])]
    rdii_lines = [(line, toks) for line, toks in rdii_lines if toks]
    if not hyd_lines and not rdii_lines:
        return None
    # SWMM's object names are not case sensitive (hash.c `samestr`); the first spelling of a
    # unit hydrograph is its name, a gage or node keeps the spelling of its own section
    canon: dict[str, str] = {}
    for _, toks in hyd_lines:
        canon.setdefault(toks[0].upper(), toks[0])
    gage_names = {g.upper(): g for g in gages}
    node_names = {n.upper(): n for n in nodes}
    order = list(canon.values())                                    # SWMM's UH numbering
    sets = {name: UnitHydrographSet.blank(name) for name in order}
    for line, toks in hyd_lines:
        name = canon[toks[0].upper()]
        s = sets[name]
        if len(toks) == 2:
            if toks[1].upper() not in gage_names:
                _fail(path, line, f"unit hydrograph {name!r} names the rain gage {toks[1]!r}, "
                                  f"which has no [RAINGAGES] entry")
            sets[name] = UnitHydrographSet(name, gage_names[toks[1].upper()], s.responses)
            continue
        if len(toks) < 6:
            _fail(path, line, "a unit hydrograph line needs a name, month, response and "
                              "R, T and K (or a name and a rain gage)")
        up = toks[1].upper()
        if up[:3] in MONTH_NAMES:
            month: int | None = MONTH_NAMES.index(up[:3]) + 1
        elif up.startswith("ALL"):
            month = None
        else:
            _fail(path, line, f"the month must be JAN to DEC or ALL, got {toks[1]!r}")
        kind = next((i for i, word in enumerate(RESPONSES) if toks[2].upper().startswith(word)),
                    None)
        if kind is None:                                # the older three-triangle line
            if len(toks) < 11:
                _fail(path, line, "a unit hydrograph line needs a response (SHORT, MEDIUM or "
                                  "LONG) and R, T and K, or nine numbers and three more for "
                                  "the abstraction")
            p = [_num(path, line, toks[i + 2], "a unit hydrograph parameter") for i in range(9)]
            ia = [_num(path, line, toks[i + 11], "an initial abstraction")
                  if len(toks) > i + 11 else 0.0 for i in range(3)]
            for k in range(3):
                s = s.with_response(
                    UnitHydrograph(p[3 * k], p[3 * k + 1], p[3 * k + 2], *ia), k, month)
            sets[name] = s
            continue
        x = [_num(path, line, toks[i + 3], "a unit hydrograph parameter") for i in range(3)]
        ia = [_num(path, line, toks[i + 6], "an initial abstraction")
              if len(toks) > i + 6 else 0.0 for i in range(3)]
        sets[name] = s.with_response(UnitHydrograph(x[0], x[1], x[2], *ia), kind, month)

    inflows: dict[str, RdiiInflow] = {}
    for line, toks in rdii_lines:
        if len(toks) < 3:
            _fail(path, line, "an RDII line needs a node, a unit hydrograph and a sewer area")
        if toks[0].upper() not in node_names:
            _fail(path, line, f"[RDII] names the node {toks[0]!r}, which is not a node")
        if toks[1].upper() not in canon:
            _fail(path, line, f"[RDII] names the unit hydrograph {toks[1]!r}, which has no "
                              f"[HYDROGRAPHS] entry")
        area = _num(path, line, toks[2], "the sewer area")
        if area < 0.0:
            _fail(path, line, f"the sewer area {toks[2]!r} is negative")
        node = node_names[toks[0].upper()]
        inflows[node] = RdiiInflow(
            node, canon[toks[1].upper()], area / UCF_LANDAREA[0 if us else 1] * LCF * LCF)
    result = Rdii(tuple(sets[n] for n in order), tuple(inflows.values()), ignore)
    if inflows and not ignore:
        try:
            result.validate()
        except ValueError as err:
            raise ValueError(f"{path}: {err}") from None
        for s in result.sets:
            if s.gage is None:
                raise ValueError(
                    f"{path}: unit hydrograph {s.name!r} has no rain gage line "
                    f"(`{s.name} <gage>` in [HYDROGRAPHS]); SWMM would read the first gage of "
                    f"the file, which is not modelled")
    return result
