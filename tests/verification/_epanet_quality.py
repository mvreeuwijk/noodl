"""The EPANET 2.2 water-quality reference of the quality parity tests, and the hydraulic
parity's (`epanet_results` and `epanet_session`, for `test_water_parity.py` and
`test_water_networks.py`).

EPANET runs through its toolkit (`wntr` 1.5.0's `ENepanet` and its ctypes handle `ENlib` on the
bundled `epanet22.dll`) on the committed file, never on `EpanetSimulator`'s rewrite of it. Its
copy (`epanet_copy`) adds only `[OPTIONS] ACCURACY 1e-8`, `TRIALS 400` (the hydraulic parity's
setting, `test_water_networks.py`) and `[REPORT] STATUS YES` (for the mass-balance section of
the report); noodl reads the unedited file. EPANET's reader clamps a file's ACCURACY to [1e-5,
1e-1] (`input3.c`, `optionvalue`), so `_open` sets it to 1e-7 after `EN_open` through
`EN_setoption(EN_ACCURACY)` (`epanet.c`, which takes 1e-8 to 1e-1; Rulings Q11, Q14: at 1e-8
EPANET is itself unbalanced on Net2 and Net3, see `_open`). Every run asserts that EPANET raised
no warning (wntr's `ENepanet` only logs codes below 100, e.g. 1, "hydraulically unbalanced").
EPANET's files live in a temporary directory, never next to the input.

The quality loop is `EN_solveQ`'s own (`epanet.c`, v2.2): `EN_openQ`, `EN_initQ(EN_NOSAVE)`,
then `EN_runQ` and `EN_nextQ` until the step is 0, reading every node's `EN_QUALITY` (12) and
every link's `EN_LINKQUAL` (14) after each `EN_runQ`: the quality at every hydraulic step
(not `EN_stepQ`, whose `stepqual` splits a quality step at a hydraulic event). Both are in
the file's quality units (`NodeQual * Ucf[QUALITY]`, `avgqual * Ucf[LINKQUAL]`): mg/L or
ug/L for a chemical, hours for AGE, percent for TRACE. The mass-balance ratio is
`EN_getstatistic(EN_MASSBALANCE = 4)` after the last `EN_nextQ` (`MassBalance.ratio`, in
full); the five terms are the report's (`report.c`, `writemassbalance`: `%12.5e`, the ratio
`%.5f`), in EPANET's internal sums (a chemical in its mass unit, AGE and TRACE in quality
times ft3); `mass_balance_si` converts them.

EPANET's quality reads its flows back from the hydraulics file as float32 (`output.c`,
`savehyd`/`readhyd`): a ~6e-8 relative floor under every comparison.
"""

from __future__ import annotations

import ctypes
import re
import tempfile
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from noodl.apps.water.units import FLOW_UNITS

F64 = torch.float64

#: `EN_NodeProperty` / `EN_LinkProperty` / `EN_CountType` / `EN_AnalysisStatistic` codes
#: (`epanet2_enums.h`, v2.2).
EN_ELEVATION, EN_HEAD, EN_QUALITY = 0, 10, 12
EN_FLOW, EN_LINKQUAL = 8, 14
EN_NODECOUNT, EN_LINKCOUNT = 0, 2
EN_MASSBALANCE = 4
EN_TRIALS, EN_ACCURACY, EN_HEADLOSSFORM = 0, 1, 7
#: `EN_HeadLossType` (`epanet2_enums.h`, v2.2): `EN_DW`.
EN_DW = 1
#: The accuracy `_open` sets (Ruling Q14; the measured cause in its docstring).
REFERENCE_ACCURACY = 1e-7
#: EPANET's litres per ft3 (`types.h`, ``LperFT3 = 28.317``; the exact value is 28.3168466).
L_PER_FT3 = 28.317
_FT3 = 0.3048**3
#: `EN_FlowUnits` in code order; the first five are US (feet).
_FLOW_UNIT_CODES = ("CFS", "GPM", "MGD", "IMGD", "AFD", "LPS", "LPM", "MLD", "CMH", "CMD")
_FOOT = 0.3048

#: The report's mass-balance lines (`writemassbalance`) and the keys they are read into.
_TERMS = {"Initial Mass:": "initial", "Mass Inflow:": "inflow", "Mass Outflow:": "outflow",
          "Mass Reacted:": "reacted", "Final Mass:": "final", "Mass Ratio:": "ratio"}
_HEADER = re.compile(r"^\s*\[([A-Za-z_]+)\]")
_END = re.compile(r"^\s*\[END\]", re.IGNORECASE)

#: What `epanet_copy` adds, before `[END]`.
_REFERENCE_SETTINGS = "[OPTIONS]\n ACCURACY 1e-8\n TRIALS 400\n[REPORT]\n STATUS YES\n"


def edited(source: Path, tmp_path: Path, name: str, *,
           add: Mapping[str, Sequence[str]] | None = None,
           replace: Sequence[tuple[str, str]] = ()) -> Path:
    """`source` edited into `tmp_path / name` (noodl's input; EPANET runs `epanet_copy` of
    it): each `(old, new)` of `replace` a literal replacement that must match, then the lines
    of `add` appended to the end of their sections (by name, any case), a missing section
    created before `[END]` (at the end of a file without one)."""
    text = Path(source).read_text()
    for old, new in replace:
        assert old in text, f"{source}: {old!r} is not in the file"
        text = text.replace(old, new)
    lines = text.splitlines()
    for section, extra in (add or {}).items():
        headers = [k for k, line in enumerate(lines)
                   if (m := _HEADER.match(line)) and m.group(1).upper() == section.upper()]
        if headers:
            start = headers[-1]
            end = next((k for k in range(start + 1, len(lines)) if _HEADER.match(lines[k])),
                       len(lines))
            lines[end:end] = list(extra)
        else:
            end = next((k for k, line in enumerate(lines) if _END.match(line)), len(lines))
            lines[end:end] = [f"[{section.upper()}]", *extra]
    out = Path(tmp_path) / name
    out.write_text("\n".join(lines) + "\n")
    return out


def epanet_copy(path: Path, directory: Path) -> Path:
    """EPANET's copy of `path` in `directory`: the file with `[OPTIONS] ACCURACY 1e-8` (which
    EPANET's reader clamps to 1e-5; `_open` sets 1e-7), `TRIALS 400` and `[REPORT] STATUS
    YES` inserted before its `[END]` (appended when it has none), and nothing else changed."""
    text = Path(path).read_text()
    found = list(re.finditer(r"^[ \t]*\[END\]", text, re.IGNORECASE | re.MULTILINE))
    if found:
        at = found[-1].start()
        text = text[:at] + _REFERENCE_SETTINGS + text[at:]
    else:
        text = text + ("" if text.endswith("\n") else "\n") + _REFERENCE_SETTINGS
    copy = Path(directory) / f"{Path(path).stem}.epanet.inp"
    copy.write_text(text)
    return copy


@dataclass
class QualityReference:
    """EPANET's quality of one network: `node` and `link` `(T, n)` in the file's quality
    units at every `EN_runQ` time of `times` (s), columns in EPANET's index order
    (`node_names`: junctions, then reservoirs and tanks, in file order; `link_names`: file
    order across `[PIPES] [PUMPS] [VALVES]`); `ratio` the mass-balance ratio in full and
    `terms` the report's `initial`, `inflow`, `outflow`, `reacted`, `final` and `ratio`."""

    times: list[int]
    node_names: list[str]
    link_names: list[str]
    node: torch.Tensor
    link: torch.Tensor
    ratio: float
    terms: dict[str, float]


def _toolkit():
    from wntr.epanet.toolkit import ENepanet
    return ENepanet()


def _open(en, copy: Path, report: Path, binary: Path, *,
          accuracy: float | None = REFERENCE_ACCURACY) -> None:
    """`EN_open` of the reference `copy`, then `EN_setoption(EN_ACCURACY, accuracy)` (the
    file's 1e-8 is clamped to 1e-5 on reading; module docstring); `accuracy=None` leaves
    the copy's clamped 1e-5, for a hydraulic reference that names its measured cause
    (`epanet_results`). The project is closed if the option is refused.

    Why 1e-7 and not 1e-8 (Ruling Q14), MEASURED (EPANET warning 1, hydraulically unbalanced,
    at the steps listed): at 1e-8 with TRIALS 400, Net2 at t = 25200, 28800 and 82800 s (411
    iterations each) and Net3 at t = 3600 and 336761 s; at 1e-8 with TRIALS 5000, Net2 none,
    Net3 still 3600 and 336761 s (it never converges there); at 1e-7 and 1e-6 with TRIALS 400,
    none on Net1, Net2, Net3 or `quality_base.inp`. A reference EPANET itself flags as
    unbalanced is no reference; 1e-7 is still 100 times tighter than a file can give."""
    en.ENopen(str(copy), str(report), str(binary))
    if accuracy is None:
        return
    code = en.ENlib.EN_setoption(en._project, EN_ACCURACY, ctypes.c_double(accuracy))
    if code != 0:
        en.ENclose()
        raise AssertionError(f"EN_setoption(EN_ACCURACY, {accuracy}) returned {code}")


def _no_warnings(en, path) -> None:
    """EPANET's warnings (codes 1 to 6) are only logged by wntr: refuse a run that had one."""
    assert not en.Warnflag, f"EPANET warned on {path}: {en.errcodelist}"


#: EPANET's warnings that describe the network's state, not the solve (`report.c`,
#: `writehydwarn`): 3 disconnected, 4 a pump shut or beyond its curve, 5 a valve that cannot
#: deliver, 6 negative pressures. 1 (unbalanced) and 2 (unstable: converged only with the
#: statuses held) say the reference is no solution and are never permitted.
NETWORK_STATE_WARNINGS = frozenset({3, 4, 5, 6})
#: The report's warning lines (`text.h`: `writehydwarn`'s per-step WARN01 to WARN06, and
#: the WARN1 to WARN6 summary) and their codes, read from the report as well as from wntr:
#: `writehydwarn` returns one flag per step, the last it set (2, 6, 5, 4, 1 in that order),
#: so an earlier warning hides behind a later one there. WARN05 starts with the valve's type.
_REPORT_WARNINGS = (
    ("WARNING: System unbalanced", 1), ("WARNING: System hydraulically unbalanced", 1),
    ("WARNING: Maximum trials exceeded", 2), ("WARNING: System may be hydraulically", 2),
    ("WARNING: Node ", 3), ("WARNING: System disconnected", 3),
    ("additional nodes disconnected", 3), ("WARNING: Pump", 4),
    ("WARNING: Valves cannot deliver", 5),
    *((f"WARNING: {kind} ", 5) for kind in ("PRV", "PSV", "PBV", "FCV", "TCV", "GPV")),
    ("WARNING: Negative pressures", 6), ("WARNING: System has negative pressures", 6),
)


def _report_warning_codes(report: str) -> list[int]:
    """The code (1 to 6) of every `WARNING` line of an EPANET report; an unknown one fails."""
    codes = []
    for line in (ln.strip() for ln in report.splitlines()):
        if "WARNING" not in line:
            continue
        found = {c for text, c in _REPORT_WARNINGS
                 if line.startswith(text) or (not text.startswith("WARNING") and text in line)}
        assert len(found) == 1, f"unrecognised EPANET report warning {line!r}"
        codes.append(found.pop())
    return codes


def _warning_codes(en) -> list[int]:
    """The codes (1 to 6) of the warnings wntr logged on `en` (`ENgetwarning`'s texts)."""
    from wntr.epanet.toolkit import EN_ERROR_CODES
    codes = []
    for text in en.errcodelist:
        found = [c for c in range(1, 7) if text.endswith(EN_ERROR_CODES[c].split("%s", 1)[1])]
        assert len(found) == 1, f"unrecognised EPANET warning {text!r}"
        codes.append(found[0])
    return codes


def option(en, code: int) -> float:
    """`EN_getoption(code)` of an open project."""
    value = ctypes.c_double()
    assert en.ENlib.EN_getoption(en._project, code, ctypes.byref(value)) == 0
    return value.value


def _names(en) -> tuple[list[str], list[str]]:
    nodes = [en.ENgetnodeid(k) for k in range(1, en.ENgetcount(EN_NODECOUNT) + 1)]
    links = []
    for k in range(1, en.ENgetcount(EN_LINKCOUNT) + 1):
        buffer = ctypes.create_string_buffer(64)
        code = en.ENlib.EN_getlinkid(en._project, k, buffer)
        assert code == 0, f"EN_getlinkid({k}) returned {code}"
        links.append(buffer.value.decode("utf-8"))
    return nodes, links


def mass_balance_terms(report: str) -> dict[str, float]:
    """The mass-balance section of an EPANET report (`writemassbalance`), `{}` without one."""
    terms = {}
    for line in report.splitlines():
        for label, key in _TERMS.items():
            if line.strip().startswith(label):
                terms[key] = float(line.split(":", 1)[1])
    return terms


def mass_balance_si(terms: Mapping[str, float], mode: str) -> dict[str, float]:
    """The report's mass-balance `terms` (`writemassbalance`'s internal sums) in quality
    units times m3, the ratio unchanged: a CHEMICAL's mass is `C * V(ft3) * LperFT3` (in the
    chemical's mass unit, `LperFT3 = 28.317`), so `/ 28.317 * ft3`; AGE and TRACE are
    `quality * V(ft3)`, so `* ft3` (`quality.c`, `evalmassbalance`; `report.c`). For a
    chemical in mg/L, mg/L * m3 is grams."""
    scale = _FT3 / L_PER_FT3 if mode == "CHEMICAL" else _FT3
    return {key: value if key == "ratio" else value * scale for key, value in terms.items()}


def epanet_quality(path: Path) -> QualityReference:
    """EPANET 2.2's quality on `path` (`EN_solveH`, then `EN_solveQ`'s `EN_runQ`/`EN_nextQ`
    loop; module docstring), in a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="epanet-q-", ignore_cleanup_errors=True) as tmp:
        directory = Path(tmp)
        copy = epanet_copy(path, directory)
        report = directory / "quality.rpt"
        en = _toolkit()
        _open(en, copy, report, directory / "quality.bin")
        try:
            node_names, link_names = _names(en)
            en.ENsolveH()
            en.ENopenQ()
            en.ENinitQ(0)
            times, node, link = [], [], []
            while True:
                times.append(int(en.ENrunQ()))
                node.append([en.ENgetnodevalue(k, EN_QUALITY)
                             for k in range(1, len(node_names) + 1)])
                link.append([en.ENgetlinkvalue(k, EN_LINKQUAL)
                             for k in range(1, len(link_names) + 1)])
                if en.ENnextQ() <= 0:
                    break
            ratio = ctypes.c_double()
            code = en.ENlib.EN_getstatistic(en._project, EN_MASSBALANCE, ctypes.byref(ratio))
            assert code == 0, f"EN_getstatistic(EN_MASSBALANCE) returned {code}"
            en.ENcloseQ()
            _no_warnings(en, path)
        finally:
            en.ENclose()
        terms = mass_balance_terms(report.read_text(errors="replace"))
    return QualityReference(
        times=times, node_names=node_names, link_names=link_names,
        node=torch.tensor(node, dtype=F64), link=torch.tensor(link, dtype=F64),
        ratio=float(ratio.value), terms=terms)


def epanet_hydraulics(path: Path) -> dict:
    """EPANET 2.2's hydraulics on `path` at every `EN_runH` time, in SI: `{"times",
    "node_names", "link_names", "head" (T, n) m, "flow" (T, m) m3/s}`."""
    with tempfile.TemporaryDirectory(prefix="epanet-h-", ignore_cleanup_errors=True) as tmp:
        directory = Path(tmp)
        copy = epanet_copy(path, directory)
        en = _toolkit()
        _open(en, copy, directory / "h.rpt", directory / "h.bin")
        try:
            node_names, link_names = _names(en)
            units = _FLOW_UNIT_CODES[en.ENgetflowunits()]
            length = _FOOT if units in _FLOW_UNIT_CODES[:5] else 1.0
            en.ENopenH()
            en.ENinitH(0)
            times, head, flow = [], [], []
            while True:
                times.append(int(en.ENrunH()))
                head.append([en.ENgetnodevalue(k, EN_HEAD) * length
                             for k in range(1, len(node_names) + 1)])
                flow.append([en.ENgetlinkvalue(k, EN_FLOW) * FLOW_UNITS[units]
                             for k in range(1, len(link_names) + 1)])
                if en.ENnextH() <= 0:
                    break
            en.ENcloseH()
            _no_warnings(en, path)
        finally:
            en.ENclose()
    return {"times": times, "node_names": node_names, "link_names": link_names,
            "head": torch.tensor(head, dtype=F64), "flow": torch.tensor(flow, dtype=F64)}


@contextmanager
def epanet_session(path: Path) -> Iterator:
    """wntr's `ENepanet` open on EPANET's copy of `path` (`epanet_copy`, `_open`: ACCURACY
    `REFERENCE_ACCURACY`) in a temporary directory, for a caller that drives the toolkit
    itself (`ENopenH`, `ENrunH`, ...); on leaving, `_no_warnings` (unless the body raised)
    and `ENclose`."""
    with tempfile.TemporaryDirectory(prefix="epanet-s-", ignore_cleanup_errors=True) as tmp:
        directory = Path(tmp)
        copy = epanet_copy(path, directory)
        en = _toolkit()
        _open(en, copy, directory / "session.rpt", directory / "session.bin")
        try:
            yield en
            _no_warnings(en, path)
        finally:
            en.ENclose()


def epanet_results(path: Path, *, duration: float | None = None, reader=None,
                   accuracy: float | None = REFERENCE_ACCURACY, trials: int | None = None,
                   warnings: Collection[int] = ()):
    """EPANET 2.2's results on `path`, the hydraulic parity's reference: what wntr's
    `EpanetSimulator.run_sim` returns (`EN_solveH`, `EN_solveQ`, the binary output read by
    wntr's `BinFile`, in SI at the report times), but on EPANET's copy of the file itself
    (`epanet_copy`), not on wntr's rewrite of it, and at a true ACCURACY `accuracy` (`_open`).
    A file's ACCURACY is clamped to 1e-5, which `accuracy=None` keeps, and `trials` replaces
    the copy's TRIALS 400 (`EN_TRIALS`): a test that passes either says why. Refused if
    EPANET warned, but for the network-state warnings `warnings` names, each of which must
    occur (`NETWORK_STATE_WARNINGS`: the case is about a pump shut by its head, or a
    negative pressure), and always on an unconverged or unstable step; every warning line of
    the report counts too (`_REPORT_WARNINGS`). `duration` (s) sets `[TIMES] DURATION` in
    the copy; `reader` is the `BinFile` (or a subclass keeping more of the file, e.g. its
    energy section) that reads the output."""
    assert set(warnings) <= NETWORK_STATE_WARNINGS, warnings
    from wntr.epanet.io import BinFile
    with tempfile.TemporaryDirectory(prefix="epanet-r-", ignore_cleanup_errors=True) as tmp:
        directory = Path(tmp)
        source = Path(path)
        if duration is not None:
            source = edited(source, directory, f"{source.stem}.duration.inp",
                            add={"TIMES": [f" DURATION {int(duration)} SEC"]})
        copy = epanet_copy(source, directory)
        binary, report = directory / "results.bin", directory / "results.rpt"
        en = _toolkit()
        _open(en, copy, report, binary, accuracy=accuracy)
        try:
            if trials is not None:
                code = en.ENlib.EN_setoption(en._project, EN_TRIALS, ctypes.c_double(trials))
                assert code == 0, f"EN_setoption(EN_TRIALS, {trials}) returned {code}"
            en.ENsolveH()
            en.ENsolveQ()
            darcy_weisbach = option(en, EN_HEADLOSSFORM) == EN_DW
        finally:
            en.ENclose()
        text = report.read_text(errors="replace")
        codes = set(_warning_codes(en)) | set(_report_warning_codes(text))
        unexpected = sorted(codes - set(warnings))
        lines = [ln.strip() for ln in text.splitlines() if "WARNING" in ln]
        assert not unexpected, (f"EPANET warned ({unexpected}) on {path}: {en.errcodelist} "
                                f"{lines[:3]}")
        absent = set(warnings) - codes
        assert not absent, f"EPANET did not give the warnings {sorted(absent)} on {path}"
        return (reader or BinFile()).read(str(binary), False, darcy_weisbach)


def run_quality(path: Path, **water_run_kwargs):
    """noodl on `path`, unedited: `read_epanet_inp`, `build_model`, `water_run(quality=True,
    diagnostics=d)`; `(results, d)`, `d` also carrying the model's `node_names` and
    `link_names` for `assert_matches_epanet`."""
    from noodl.apps.water.inp import read_epanet_inp
    from noodl.apps.water.network import build_model
    from noodl.apps.water.simulate import water_run
    model, _, _ = build_model(read_epanet_inp(path))
    diagnostics: dict = {}
    results = water_run(model, quality=True, diagnostics=diagnostics, **water_run_kwargs)
    diagnostics["node_names"] = list(model.node_names)
    diagnostics["link_names"] = list(model.link_names)
    return results, diagnostics


def first_divergence(times, ours, theirs, names, *, limit: float):
    """The earliest step at which `|ours - theirs|` (`(T, n)`) exceeds `limit` somewhere:
    `(time, name, difference)` of the worst column at that step, `None` if no step does."""
    error = (torch.as_tensor(ours, dtype=F64) - torch.as_tensor(theirs, dtype=F64)).abs()
    for row in range(error.shape[0]):
        worst = int(torch.argmax(error[row]))
        if float(error[row, worst]) > limit:
            return times[row], names[worst], float(error[row, worst])
    return None


def assert_matches_epanet(diagnostics: dict, reference: QualityReference, *, ctol: float,
                          admitted: Sequence[tuple[int, str]] = (), instance: int = 0) -> dict:
    """noodl's quality (`diagnostics["quality"][instance]`: `"times"` s, `"node_quality"` and
    `"link_quality"` `(T, n)` in the file's quality units, columns named by
    `diagnostics["node_names"]` and `["link_names"]`, else in the reference's order; the same
    names as EPANET's) against EPANET's at every one of EPANET's times (noodl may have more),
    over the whole (time, node and link) matrix (the spec's Verification bounds): with `scale =
    max(|reference|, ctol)`, the worst difference within `ctol` and every point beyond `1e-5 *
    scale` one of `admitted` (`(time, name)`, a node or link of that name; the spec admits one
    only with its flipped merge written in the test). That per-point rule is the spec's "99.9th
    percentile within 1e-5 of the scale" made strict: at most the admitted points exceed it,
    whatever the matrix's size (on a matrix under a thousand points the 99.9th percentile
    interpolates between its two largest differences, so a single admitted point would decide
    it; Ruling Q13). A failure names `first_divergence`. Returns `{"p999"` (of the whole matrix,
    the spec's measure), `"worst", "scale", "first"}`."""
    ours = diagnostics["quality"][instance]
    node_names = list(diagnostics.get("node_names", reference.node_names))
    link_names = list(diagnostics.get("link_names", reference.link_names))
    assert sorted(node_names) == sorted(reference.node_names), "the nodes differ from EPANET's"
    assert sorted(link_names) == sorted(reference.link_names), "the links differ from EPANET's"
    nodes = [reference.node_names.index(n) for n in node_names]
    links = [reference.link_names.index(n) for n in link_names]
    our_times = [round(float(t)) for t in ours["times"]]
    common = [t for t in reference.times if t in our_times]
    missing = [t for t in reference.times if t not in our_times]
    assert not missing, (f"noodl's quality covers {len(common)} of EPANET's "
                         f"{len(reference.times)} times; missing {missing[:5]}")
    mine = [our_times.index(t) for t in common]
    rows = [reference.times.index(t) for t in common]
    computed = torch.cat([torch.as_tensor(ours["node_quality"], dtype=F64)[mine],
                          torch.as_tensor(ours["link_quality"], dtype=F64)[mine]], dim=1)
    expected = torch.cat([reference.node[rows][:, nodes], reference.link[rows][:, links]], dim=1)
    assert computed.shape == expected.shape, (computed.shape, expected.shape)
    assert bool(torch.isfinite(computed).all()), "noodl's quality is not finite"
    names = node_names + link_names
    error = (computed - expected).abs()
    scale = max(float(expected.abs().max()), ctol)
    limit = 1e-5 * scale
    p999 = float(np.quantile(error.numpy(), 0.999))
    worst = float(error.max())
    first = first_divergence(common, computed, expected, names, limit=limit)
    allowed = {(int(t), str(n)) for t, n in admitted}
    beyond = [(common[i], names[j]) for i, j in (error > limit).nonzero().tolist()]
    unadmitted = [point for point in beyond if point not in allowed]
    found = {"p999": p999, "worst": worst, "scale": scale, "first": first}
    if unadmitted or worst > ctol:
        at = np.unravel_index(int(torch.argmax(error)), tuple(error.shape))
        diverged = ("none beyond 1e-5 of the scale" if first is None else
                    f"first divergence at {first[0]} s at {first[1]} ({first[2]:.3e} > "
                    f"{limit:.3e} = 1e-5 of the scale {scale:.4g})")
        raise AssertionError(
            f"quality differs from EPANET: {diverged}; worst "
            f"{worst:.3e} (Ctol {ctol:.3e}) at {common[at[0]]} s at {names[at[1]]}; "
            f"99.9th percentile {p999:.3e}; {len(unadmitted)} point(s) beyond 1e-5 of the "
            f"scale not admitted, the first {unadmitted[:3]}")
    return found
