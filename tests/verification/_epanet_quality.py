"""The EPANET 2.2 water-quality reference of the quality parity tests.

EPANET runs through its toolkit (`wntr` 1.5.0's `ENepanet` and its ctypes handle `ENlib` on
the bundled `epanet22.dll`) on the committed file, never on `EpanetSimulator`'s rewrite of
it. Its copy (`epanet_copy`) adds only `[OPTIONS] ACCURACY 1e-8`, `TRIALS 400` (the hydraulic
parity's setting, `test_water_networks.py`) and `[REPORT] STATUS YES` (for the mass-balance
section of the report); noodl reads the unedited file. EPANET's reader clamps a file's
ACCURACY to [1e-5, 1e-1] (`input3.c`, `optionvalue`; only `EN_setoption` takes 1e-8), so the
copy runs at 1e-5, as do the hydraulic parity's copies. EPANET's files live in a temporary
directory, never next to the input.

The quality loop is `EN_solveQ`'s own (`epanet.c`, v2.2): `EN_openQ`, `EN_initQ(EN_NOSAVE)`,
then `EN_runQ` and `EN_nextQ` until the step is 0, reading every node's `EN_QUALITY` (12) and
every link's `EN_LINKQUAL` (14) after each `EN_runQ`: the quality at every hydraulic step
(not `EN_stepQ`, whose `stepqual` splits a quality step at a hydraulic event). Both are in
the file's quality units (`NodeQual * Ucf[QUALITY]`, `avgqual * Ucf[LINKQUAL]`): mg/L or
ug/L for a chemical, hours for AGE, percent for TRACE. The mass-balance ratio is
`EN_getstatistic(EN_MASSBALANCE = 4)` after the last `EN_nextQ` (`MassBalance.ratio`, in
full); the five terms are the report's (`report.c`, `writemassbalance`: `%12.5e`, the ratio
`%.5f`), in EPANET's internal sums (a chemical in its mass unit, AGE and TRACE in quality
times ft3).

EPANET's quality reads its flows back from the hydraulics file as float32 (`output.c`,
`savehyd`/`readhyd`): a ~6e-8 relative floor under every comparison.
"""

from __future__ import annotations

import ctypes
import re
import tempfile
from collections.abc import Mapping, Sequence
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


def epanet_copy(path: Path, directory: Path | None = None) -> Path:
    """EPANET's copy of `path` in `directory` (a fresh temporary directory when `None`): the
    file with `[OPTIONS] ACCURACY 1e-8` (read as 1e-5, module docstring), `TRIALS 400` and
    `[REPORT] STATUS YES` inserted before its `[END]` (appended when it has none), and nothing
    else changed."""
    text = Path(path).read_text()
    found = list(re.finditer(r"^[ \t]*\[END\]", text, re.IGNORECASE | re.MULTILINE))
    if found:
        at = found[-1].start()
        text = text[:at] + _REFERENCE_SETTINGS + text[at:]
    else:
        text = text + ("" if text.endswith("\n") else "\n") + _REFERENCE_SETTINGS
    directory = Path(tempfile.mkdtemp(prefix="epanet-")) if directory is None else directory
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


def epanet_quality(path: Path) -> QualityReference:
    """EPANET 2.2's quality on `path` (`EN_solveH`, then `EN_solveQ`'s `EN_runQ`/`EN_nextQ`
    loop; module docstring), in a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="epanet-q-", ignore_cleanup_errors=True) as tmp:
        directory = Path(tmp)
        copy = epanet_copy(path, directory)
        report = directory / "quality.rpt"
        en = _toolkit()
        en.ENopen(str(copy), str(report), str(directory / "quality.bin"))
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
        en.ENopen(str(copy), str(directory / "h.rpt"), str(directory / "h.bin"))
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
        finally:
            en.ENclose()
    return {"times": times, "node_names": node_names, "link_names": link_names,
            "head": torch.tensor(head, dtype=F64), "flow": torch.tensor(flow, dtype=F64)}


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
    """noodl's quality (`diagnostics["quality"][instance]`: `"times"` s, `"node_quality"`
    and `"link_quality"` `(T, n)` in the file's quality units, columns named by
    `diagnostics["node_names"]` and `["link_names"]`, else in the reference's order) against
    EPANET's at the common times, over the whole (time, node and link) matrix (the spec's
    Verification bounds): with `scale = max(|reference|, ctol)`, the worst difference within
    `ctol` and every point beyond `1e-5 * scale` one of `admitted` (`(time, name)`, a node or
    link of that name; the spec admits one only with its flipped merge written in the test).
    That per-point rule is the spec's "99.9th percentile within 1e-5 of the scale" made
    strict: at most the admitted points exceed it, whatever the matrix's size (a 99.9th
    percentile of a matrix under a thousand points is its worst). A failure names
    `first_divergence`. Returns `{"p999"` (of the whole matrix, the spec's measure),
    `"worst", "scale", "first"}`."""
    ours = diagnostics["quality"][instance]
    node_names = list(diagnostics.get("node_names", reference.node_names))
    link_names = list(diagnostics.get("link_names", reference.link_names))
    nodes = [reference.node_names.index(n) for n in node_names]
    links = [reference.link_names.index(n) for n in link_names]
    our_times = [round(float(t)) for t in ours["times"]]
    common = [t for t in reference.times if t in our_times]
    assert common, f"no common times: noodl {our_times[:5]}..., EPANET {reference.times[:5]}..."
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
