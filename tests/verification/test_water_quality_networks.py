"""EPANET's example networks' water quality run whole through `water_run` against EPANET 2.2
(`wntr` 1.5.0's toolkit, `tests/verification/_epanet_quality.py`): Net1 (the committed
`tests/data/water/Net1.inp`, chlorine, 24 h), Net2 (fluoride, a CONCEN source on a pattern
at junction 1's negative demand, 55 h), Net3 (TRACE from the Lake, 168 h), Net6 (zero
quality as shipped; a CONCEN source at its reservoir added, 96 h), ky4 (TRACE from R-1, 24 h)
and ky10 (AGE, to 6:54:59, where its hydraulics leave EPANET's), `wntr`'s files as shipped
unless a test says otherwise (ky4's and ky10's DURATION 0 set to 24 h,
`test_water_networks._ky_24h`).

The bounds are the spec's (Verification): node and link quality over the whole (time, node
and link) matrix, every point within 1e-5 of the quality's scale (`assert_matches_epanet`,
the 99.9th percentile made strict, Ruling Q13) and the worst within `Ctol`; the mass balance
(`_mass_balance`) with every term within 2e-5 relative and the ratio within 1e-6. Each test
names its MEASURED numbers beside its bounds; a bound that differs, or a point admitted beyond
them, has its measured cause in the test. Net2, Net3, Net6, ky4 and ky10 are `slow`.
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.water import simulate
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model
from noodl.apps.water.quality import EXACT_LITRES_PER_M3, QualityRouter, quality_step
from noodl.apps.water.simulate import water_run
from noodl.apps.water.units import EPANET_QCF, FLOW_UNITS, rounded_cfs_per_m3s

wntr = pytest.importorskip("wntr")

from tests.verification._epanet_quality import (  # noqa: E402
    EN_ACCURACY,
    QualityReference,
    assert_matches_epanet,
    edited,
    epanet_hydraulics,
    epanet_quality,
    epanet_results,
    epanet_session,
    first_divergence,
    flow_units,
    mass_balance_si,
    node_and_link_ids,
    run_quality,
)
from tests.verification.test_water_networks import _KY_EPANET, _ky_24h  # noqa: E402

F64 = torch.float64
DATA = Path(__file__).resolve().parents[1] / "data" / "water"
LIBRARY = Path(wntr.__file__).parent / "library" / "networks"
NET2 = LIBRARY / "Net2.inp"


def _mass_balance(path: Path, results, reference: QualityReference) -> tuple[dict, float]:
    """Each mass-balance term's difference over EPANET's (`test_water_quality.
    _mass_balance_differences`'s rule: noodl's physical mass against the report's terms in
    quality x m3, a CHEMICAL's times exactly 1000 L/m3, EPANET's rounded `LperFT3` removed,
    Ruling Q28; 0 when both are 0) and the ratio's absolute difference."""
    mode = read_epanet_inp(path).quality.mode
    theirs = mass_balance_si(reference.terms, mode)
    unit = EXACT_LITRES_PER_M3 if mode == "CHEMICAL" else 1.0
    ours = results.mass_balance
    relative = {}
    for key in ("initial", "inflow", "outflow", "reacted", "final"):
        mine, expected = float(getattr(ours, key)), theirs[key] * unit
        relative[key] = (0.0 if mine == expected == 0.0 else
                         abs(mine - expected) / max(abs(expected), 1e-300))
    return relative, abs(float(ours.ratio) - reference.ratio)


def _assert_mass_balance(path: Path, results, reference: QualityReference, *,
                         terms: float = 2e-5, ratio: float = 1e-6) -> dict:
    relative, ratio_difference = _mass_balance(path, results, reference)
    assert ratio_difference < ratio, (float(results.mass_balance.ratio), reference.ratio)
    for key, value in relative.items():
        assert value < terms, (key, float(getattr(results.mass_balance, key)), value)
    return relative


#: `EN_NodeProperty` / `EN_LinkProperty` / `EN_TimeParameter` codes (`epanet2_enums.h`):
#: `EN_DEMAND`, `EN_FLOW`, `EN_STATUS`; `EN_DURATION`, `EN_HYDSTEP`.
_EN_DEMAND, _EN_FLOW, _EN_STATUS, _EN_DURATION, _EN_HYDSTEP = 9, 8, 11, 0, 1


def _router_on_epanets_hydraulics(path: Path, *, trials: int | None = None) -> dict:
    """noodl's `QualityRouter` driven by EPANET's own hydraulics on `path` instead of
    noodl's: at every `EN_runH` time the quality is recorded, then the router advances over
    EPANET's step (`EN_nextH`; only while `t + step` is within the duration, `nextqual`)
    with EPANET's link flows, CLOSED states and junction demands (`NodeDemand`) as its
    quality reads them back from its hydraulics file (float32 in cfs, `savehyd`; closed
    links 0), in m3/s; `trials` as for `epanet_quality`. The diagnostics, named as
    `assert_matches_epanet` takes them."""
    net = read_epanet_inp(path)
    model, _, _ = build_model(net)
    diagnostics: dict = {"node_names": list(model.node_names),
                         "link_names": list(model.link_names)}
    with epanet_session(path, trials=trials) as en:
        units = flow_units(en)
        qcf, cfs = EPANET_QCF[units], rounded_cfs_per_m3s(units)
        nodes, links = node_and_link_ids(en)
        link_index = [links.index(n) + 1 for n in model.link_names]
        junction_index = [nodes.index(j.name) + 1 for j in net.junctions]
        duration = int(en.ENgettimeparam(_EN_DURATION))
        router = QualityRouter(
            model, net.quality, inputs={}, attached=False, diagnostics=diagnostics,
            quality_step=quality_step(net, int(en.ENgettimeparam(_EN_HYDSTEP))))

        def saved(value):       # as `savehyd` keeps it, in m3/s
            return float(np.float32(value / qcf)) / cfs

        en.ENopenH()
        en.ENinitH(0)
        while True:
            t = int(en.ENrunH())
            flow = torch.tensor([saved(en.ENgetlinkvalue(k, _EN_FLOW)) for k in link_index],
                                dtype=F64)
            closed = torch.tensor([en.ENgetlinkvalue(k, _EN_STATUS) == 0 for k in link_index])
            demand = torch.tensor(
                [saved(en.ENgetnodevalue(k, _EN_DEMAND)) for k in junction_index], dtype=F64)
            router.record(t)
            step = int(en.ENnextH())
            if step <= 0:
                break
            if t + step <= duration:
                router.advance(t, step, flow=flow, closed=closed, demand=demand)
        en.ENcloseH()
    return diagnostics


def _compare(path: Path):
    """noodl and EPANET on `path`, compared by the spec's bounds: `(results, diagnostics,
    reference, found)`."""
    results, diagnostics = run_quality(path)
    reference = epanet_quality(path)
    found = assert_matches_epanet(diagnostics, reference,
                                  ctol=read_epanet_inp(path).quality.tolerance)
    return results, diagnostics, reference, found


def test_net1_quality_matches_epanet():
    """Net1 (committed), chlorine with bulk and wall decay, 24 h, at every one of EPANET's 27
    quality times. MEASURED: worst 1.6e-6 mg/L (pipe 11 at 16 h; scale 1 mg/L), 99.9th
    percentile 1.4e-6, none beyond 1e-5 of the scale; mass-balance terms within 7.7e-7
    relative, the ratio 1.8e-8 from EPANET's; flows reversed at 45154 and 81690 s."""
    path = DATA / "Net1.inp"
    results, diagnostics, reference, found = _compare(path)
    assert found["worst"] < 1e-5 * found["scale"]
    assert len(reference.times) == 27
    assert diagnostics["quality"][0]["reversals"]
    _assert_mass_balance(path, results, reference)


@pytest.mark.slow
def test_net2_hydraulics_match_epanet():
    """Net2's hydraulics, which its quality runs on (55 h, hourly, no tank or control event
    between): the hydraulic step plan is EPANET's (`EN_runH` times), and at every report
    time heads within 1e-5 relative (to `max(|H|, 1 m)`), the tank level within 2e-4 m and
    flows within 1e-6 m3/s of EPANET's at ACCURACY 1e-7 (`epanet_results`), states equal
    (`test_water_networks`'s bounds). MEASURED: heads 1.1e-7, level 8.9e-6 m, flows 1.9e-8
    m3/s, states equal, the 56 step times equal."""
    reference = epanet_results(NET2)
    steps = epanet_hydraulics(NET2)
    net = read_epanet_inp(NET2)
    model, _, _ = build_model(net)
    diagnostics: dict = {}
    run = water_run(model, quality=False, diagnostics=diagnostics)
    assert [round(t) for t in diagnostics["times"]] == steps["times"]
    names, links = model.node_names, model.link_names
    assert run.time.tolist() == [float(t) for t in reference.node["head"].index]
    heads = torch.tensor(reference.node["head"][names].to_numpy(), dtype=F64)
    worst_h = float(((run.head - heads).abs() / heads.abs().clamp(min=1.0)).max())
    worst_l = max(float((run.tank_level[:, k] - (heads[:, names.index(t.name)]
                                                - t.elevation)).abs().max())
                  for k, t in enumerate(net.tanks))
    flows = torch.tensor(reference.link["flowrate"][links].to_numpy(), dtype=F64)
    worst_q = float((run.flow - flows).abs().max())
    status = torch.tensor(reference.link["status"][links].to_numpy()).to(run.status.dtype)
    assert bool((run.status == status).all())
    assert worst_h < 1e-5, worst_h
    assert worst_l < 2e-4, worst_l
    assert worst_q < 1e-6, worst_q


def _net2_tolerance(tmp_path: Path, tolerance: str) -> Path:
    """Net2 with `[OPTIONS] Tolerance` (`Ctol`) set to `tolerance` mg/L, nothing else changed."""
    return edited(NET2, tmp_path, f"Net2-ctol-{tolerance}.inp",
                  replace=[(" Tolerance          \t0.01", f" Tolerance          \t{tolerance}")])


@pytest.mark.slow
@pytest.mark.parametrize("tolerance", ["0.00999999", "0.01000001"])
def test_net2_quality_matches_epanet(tmp_path, tolerance):
    """Net2: fluoride from a CONCEN source of 1 mg/L on pattern 3 at junction 1, whose
    demand is negative (the pump station's outflow, pattern 2, which is 0 for hours at a
    time), 55 h, every one of EPANET's 56 quality times, with `Ctol` moved 1e-8 mg/L off the
    file's 0.01 to either side (the file's own 0.01 is a merge tie, measured in
    `test_net2_as_shipped_differs_from_epanet_only_at_its_ctol_ties`). MEASURED (both
    sides): worst 4.2e-6 mg/L (scale 1.05 mg/L), 99.9th percentile 6.6e-7, none beyond 1e-5
    of the scale; mass-balance terms within the spec's bounds."""
    path = _net2_tolerance(tmp_path, tolerance)
    results, diagnostics, reference, found = _compare(path)
    assert found["worst"] < 1e-5 * found["scale"]
    assert len(reference.times) == 56
    _assert_mass_balance(path, results, reference)


def _until(reference: QualityReference, end: int) -> QualityReference:
    """`reference` at its times before `end` (s) only."""
    rows = [k for k, t in enumerate(reference.times) if t < end]
    return QualityReference(times=[reference.times[k] for k in rows],
                            node_names=reference.node_names, link_names=reference.link_names,
                            node=reference.node[rows], link=reference.link[rows],
                            ratio=reference.ratio, terms=reference.terms)


def _matrix(diagnostics: dict, reference: QualityReference) -> tuple[torch.Tensor, ...]:
    """noodl's and EPANET's (time, node and link) quality matrices at EPANET's times, in
    noodl's column order."""
    ours = diagnostics["quality"][0]
    times = [round(float(t)) for t in ours["times"]]
    rows = [times.index(t) for t in reference.times]
    mine = torch.cat([torch.as_tensor(ours["node_quality"], dtype=F64)[rows],
                      torch.as_tensor(ours["link_quality"], dtype=F64)[rows]], dim=1)
    theirs = torch.cat(
        [reference.node[:, [reference.node_names.index(n) for n in diagnostics["node_names"]]],
         reference.link[:, [reference.link_names.index(n) for n in diagnostics["link_names"]]]],
        dim=1)
    return mine, theirs


#: Net2's merge ties: pipe 1 from junction 1 (the source) carries 0.09 mg/L through hour 30
#: and takes 0.08 mg/L from 31 h (pattern 3), 0.01 mg/L apart: `Ctol` exactly; again 0.08 to
#: 0.09 at 37 h and 0.09 to 0.08 at 49 h.
_NET2_FIRST_TIE = 31 * 3600
_NET2_LAST_TIE = 49 * 3600
#: EPANET's 37 h tie, which it merges (measured at 38 h).
_NET2_EPANET_MERGES = 38 * 3600
#: Pipe 1's last segment and junction 1's outflow at the 31 h tie, as documentation only (no
#: assertion computes on them; the decisions are checked from the runs): MEASURED by hand
#: during Task 9, EPANET's in its internal unit (its outflow reproduced from
#: `findsourcequal` on its float32 demand read from `EN_savehydfile`; its segment the
#: reported 0.09000000000000001 mg/L over `Ucf`), noodl's in mg/L after the merges of hour
#: 30 (an instrumented `QualityTolerance.merges`). EPANET: `0.28317000000000014`, not below
#: `Ctol / Ucf = 0.28317`, not merged; noodl: `0.2831699999999997` in that unit, merged.
_NET2_TIE_31H = {"EPANET": (2.5485300000000004, 2.2653600000000003),
                 "noodl": (0.08999999999999998, 0.08)}


@pytest.mark.slow
def test_net2_as_shipped_differs_from_epanet_only_at_its_ctol_ties(tmp_path):
    """Net2 as shipped (`Ctol` 0.01 mg/L) matches EPANET within 1e-5 of the scale up to
    32 h and within `Ctol` everywhere; the rest is a decision at a `Ctol` tie, which the
    spec does not admit as accuracy (Ruling Q25) and this test identifies.

    MEASURED: 667 of 4256 points beyond 1e-5 of the scale, all from 32 h (115200 s), the
    first at pipe 1 (0.08283 mg/L in noodl, 0.08 in EPANET); worst 2.83e-3 mg/L there
    (791 points before the decisions moved to EPANET's unit, Ruling Q32). CAUSE: pattern 3
    steps by exactly `Ctol` at 31, 37 and 49 h, where junction 1's outflow meets pipe 1's
    last segment (`evalnodeoutflow`: merged when `|c_seg - c| < Ctol`). noodl decides in
    EPANET's internal unit (`QualityTolerance`), so a clean tie goes EPANET's way, and the
    49 h one does (0.09000000000000001 against 0.08000000000000002 mg/L: not merged in
    both). At 31 h the operands themselves differ in their last bits (`_NET2_TIE_31H`):
    EPANET's segment holds 28.317 x 0.09 rounded up by an ulp, noodl's 0.08999999999999998
    mg/L after the hour's merges of equal values (`(c v1 + c v2) / (v1 + v2)` rounds by
    the values' unit and the volumes'), so EPANET does not merge and noodl does. At 37 h
    both merge. That last bit cannot be reproduced in float64 without carrying EPANET's
    internal units through every value (MEASURED, Ruling Q32's evidence): noodl with every
    quality, source and `Ctol` given in EPANET's internal unit still flips at 31 h on its
    own hydraulics (791 points); fed EPANET's own float32 saved hydraulics in those units
    it resolves 31 h but flips at 49 h (124 points, its volumes in m3 not ft3); fed them in
    mg/L (`_router_on_epanets_hydraulics`) its 49 h segment is 0.08999999999999997 and it
    flips at 31 and 49 h (791 points). The tie follows the arithmetic's last bit; off it both
    agree within 1e-5 of the scale (`test_net2_quality_matches_epanet`, Ruling Q32's
    fallback). The test checks the decisions from the runs: EPANET at 0.01 is EPANET at
    0.01 - 1e-8 until 38 h (not merged at 31 h); noodl at 0.01 is noodl at 0.01 + 1e-8
    until 50 h (merged at 31 and 37 h), and after the first quality step of hour 31 pipe 1
    holds one segment in noodl, two where the tie is not merged; after hour 49's first two,
    as many as where it is not."""
    results, diagnostics = run_quality(NET2)
    reference = epanet_quality(NET2)
    ctol = read_epanet_inp(NET2).quality.tolerance
    assert ctol == 0.01
    # within the spec's bounds up to the first tie's step, within Ctol after
    assert_matches_epanet(diagnostics, _until(reference, _NET2_FIRST_TIE + 3600), ctol=ctol)
    mine, theirs = _matrix(diagnostics, reference)
    names = diagnostics["node_names"] + diagnostics["link_names"]
    error = (mine - theirs).abs()
    scale = float(theirs.abs().max())
    assert 2e-3 < float(error.max()) < ctol
    first = first_divergence(reference.times, mine, theirs, names, limit=1e-5 * scale)
    assert first[0] == _NET2_FIRST_TIE + 3600
    pipe = len(diagnostics["node_names"]) + diagnostics["link_names"].index("1")
    assert float(error[reference.times.index(first[0]), pipe]) == pytest.approx(
        float(error.max()), rel=1e-9)
    # the decisions, from the runs
    below, above = (_net2_tolerance(tmp_path, t) for t in ("0.00999999", "0.01000001"))
    early = [k for k, t in enumerate(reference.times) if t < _NET2_EPANET_MERGES]
    epanet_below = epanet_quality(below)
    assert torch.equal(epanet_below.node[early], reference.node[early])
    assert torch.equal(epanet_below.link[early], reference.link[early])
    assert not torch.equal(epanet_quality(above).link, reference.link)
    _, merged = run_quality(above)
    before = [k for k, t in enumerate(reference.times) if t <= _NET2_LAST_TIE]
    assert torch.equal(_matrix(merged, reference)[0][before], mine[before])
    _, split = run_quality(below)
    link = diagnostics["link_names"].index("1")
    segments = diagnostics["quality"][0]["segments"]
    step = _NET2_FIRST_TIE // 300     # the quality steps of 300 s before hour 31's first
    assert segments[step][link] == 1
    assert split["quality"][0]["segments"][step][link] == 2
    late = _NET2_LAST_TIE // 300 + 1
    assert segments[late][link] == split["quality"][0]["segments"][late][link] == 2
    assert merged["quality"][0]["segments"][late][link] == 1
    # EPANET's own saved hydraulics, fed in mg/L, flip at the same first tie
    driven = _router_on_epanets_hydraulics(NET2)
    assert first_divergence(reference.times, _matrix(driven, reference)[0], theirs, names,
                            limit=1e-5 * scale)[0] == _NET2_FIRST_TIE + 3600
    # the mass balance is within the spec's bounds even so (MEASURED: outflow 1.3e-5, final
    # 1.3e-5 relative, ratio 6.2e-9)
    _assert_mass_balance(NET2, results, reference)


@pytest.mark.slow
def test_net3_quality_matches_epanet():
    """Net3, TRACE from the Lake (percent), 168 h, every one of EPANET's 183 quality times,
    with flow reversals and re-sorts after t = 0 (pump and pipe controls, AT TIME and
    level). MEASURED: worst 2.5e-4 % (node 249 at 118 h; scale 100 %), 99.9th percentile
    9.7e-5 %, none beyond 1e-5 of the scale (1e-3 %); 1226 reversals, 154 sorts; mass
    balance within 1.4e-6 relative, the ratio 1.3e-8 from EPANET's."""
    path = LIBRARY / "Net3.inp"
    results, diagnostics, reference, found = _compare(path)
    assert found["worst"] < 1e-5 * found["scale"]
    assert len(reference.times) == 183
    quality = diagnostics["quality"][0]
    assert quality["reversals"]
    assert any(t > 0 for t, _ in quality["node_order"])
    _assert_mass_balance(path, results, reference)


NET6 = LIBRARY / "Net6.inp"
#: Net6's step plans part at 18922 s, EPANET's PUMP-3845 switch, which noodl takes at 18921 s
#: (TANK-3336 at 19 ft; `test_water_networks._NETWORKS["Net6"]`, the closed-link leak noodl
#: does not model, Ruling 30).
_NET6_PLANS_PART = 18922


def _net6_with_a_source(tmp_path: Path) -> Path:
    """Net6 with `[SOURCES] RESERVOIR-3323 CONCEN 1.0` (its only reservoir; the file's
    CHEMICAL mg/L, `Ctol` 0.01, no reactions), nothing else changed."""
    return edited(NET6, tmp_path, "Net6-source.inp",
                  add={"SOURCES": [" RESERVOIR-3323 CONCEN 1.0"]})


@pytest.mark.slow
def test_net6_quality_is_a_smoke_case():
    """Net6 as shipped: CHEMICAL with no initial quality, no source and no reaction, 96 h,
    3323 junctions, 609 hydraulic steps: it runs, every node's and link's quality is 0 at
    every step, and the mass balance has nothing in it (ratio 1, EPANET's too). MEASURED:
    all exactly 0, ratio exactly 1 in both."""
    results, diagnostics = run_quality(NET6)
    quality = diagnostics["quality"][0]
    assert len(quality["times"]) == 609
    assert not any(any(row) for row in quality["node_quality"])
    assert not any(any(row) for row in quality["link_quality"])
    assert abs(float(results.mass_balance.ratio) - 1.0) < 1e-3
    assert abs(epanet_quality(NET6).ratio - 1.0) < 1e-3


@pytest.mark.slow
def test_net6_with_an_added_source_matches_epanet(tmp_path):
    """Net6 with a CONCEN source of 1 mg/L at its reservoir (`_net6_with_a_source`), 96 h:
    `water_run` within the spec's bounds for as long as noodl's hydraulic step plan is
    EPANET's and the mass balance within them over the whole run; and noodl's router driven
    by EPANET's own saved hydraulics (`_router_on_epanets_hydraulics`) at every one of
    EPANET's 609 quality times, every point within 1e-9 of the scale (Ruling Q33).

    MEASURED, `water_run`: before 18922 s (40 steps) worst 5.9e-6 mg/L, 99.9th percentile
    7.0e-7 (scale 1 mg/L); mass-balance terms within 1.9e-6 relative, the ratio 8.8e-8 from
    EPANET's. From 18922 s the step plans part (`_NET6_PLANS_PART`; 252 of EPANET's 609 step
    times 1 to 55 s from noodl's, the source's fronts arriving at shifted times: over the
    357 times both share, worst 0.57 mg/L at JUNCTION-2292 at 107083 s, 32 % of the points
    beyond 1e-5 of the scale), Net6's hydraulic parity (levels 3e-3 m, flows 5e-4 m3/s at its
    switching times, the closed-link leak noodl does not model, Ruling 30): that span is not
    compared on `water_run`'s hydraulics. MEASURED, the router on EPANET's hydraulics: worst
    5.2e-11 mg/L, 99.9th percentile 4.7e-11 (EPANET's flows in double precision instead of
    its saved float32 give 1.8e-5 at one point, JUNCTION-1361 at 236260 s: the float32
    floor)."""
    path = _net6_with_a_source(tmp_path)
    results, diagnostics = run_quality(path)
    reference = epanet_quality(path)
    assert len(reference.times) == 609
    ours = [round(float(t)) for t in diagnostics["quality"][0]["times"]]
    parted = [t for t in reference.times if t not in ours]
    assert parted[0] == _NET6_PLANS_PART and _NET6_PLANS_PART - 1 in ours
    ctol = read_epanet_inp(path).quality.tolerance
    found = assert_matches_epanet(diagnostics, _until(reference, _NET6_PLANS_PART), ctol=ctol)
    assert found["worst"] < 1e-5 * found["scale"]
    _assert_mass_balance(path, results, reference)
    driven = assert_matches_epanet(_router_on_epanets_hydraulics(path), reference, ctol=ctol)
    assert driven["worst"] < 1e-9 * driven["scale"]


#: `EN_NodeProperty` `EN_HEAD` (`epanet2_enums.h`), for `_epanets_closed_link_leaks`.
_EN_HEAD = 10
#: EPANET's conductance of a closed link is `1 / CBIG` (`hydcoeffs.c`, CBIG = 1e8 ft per cfs).
_CBIG_FT_PER_CFS = 1e8


def _epanets_closed_link_leaks(path: Path, model, *,
                               trials: int | None = None) -> tuple[dict, set]:
    """The flow EPANET's closed links carry at every `EN_runH` time of `path` before its
    duration (the solve at the duration routes nothing), as noodl sources at the links'
    junction ends: `({t: (n_nodes,) m3/s}, the links closed at any of those times)`. A
    closed link conducts `1 / CBIG` (`hydcoeffs.c`: `P = 1 / CBIG`), so it carries
    `(H1 - H2) / CBIG` cfs, which EPANET's toolkit reports as 0 (`EN_FLOW` of a closed link)
    and its quality reads as 0 (`savehyd`). noodl's closed links carry none (Ruling 30)."""
    net = model.water_network
    ends = {link.name: (link.u, link.v) for link in (*net.pipes, *net.pumps, *net.valves)}
    junctions = {j.name for j in net.junctions}
    names = model.node_names
    leaks, closed = {}, set()
    with epanet_session(path, trials=trials) as en:
        assert flow_units(en) in ("CFS", "GPM", "MGD", "IMGD", "AFD")   # heads in ft
        nodes, links = node_and_link_ids(en)
        duration = int(en.ENgettimeparam(_EN_DURATION))
        en.ENopenH()
        en.ENinitH(0)
        while True:
            t = int(en.ENrunH())
            if t >= duration:
                break
            source = torch.zeros(len(names), dtype=F64)
            for k, name in enumerate(links, start=1):
                if en.ENgetlinkvalue(k, _EN_STATUS) != 0:
                    continue
                closed.add(name)
                u, v = ends[name]
                drop = (en.ENgetnodevalue(nodes.index(u) + 1, _EN_HEAD)
                        - en.ENgetnodevalue(nodes.index(v) + 1, _EN_HEAD))
                q = drop / _CBIG_FT_PER_CFS * 0.3048**3
                if u in junctions:
                    source[names.index(u)] -= q
                if v in junctions:
                    source[names.index(v)] += q
            leaks[t] = source
            if en.ENnextH() <= 0:
                break
        en.ENcloseH()
    return leaks, closed


def _with_epanets_closed_link_leaks(path: Path, monkeypatch, *, trials: int | None = None):
    """noodl on `path` with EPANET's closed-link leaks (`_epanets_closed_link_leaks`, EPANET's
    at the latest of its times at or before each solve) added to the junctions' sources: an
    experiment only (the run's `_Patterns.drivers_at` patched). `(diagnostics, injected,
    closed)`: the junctions a leak is injected at and the links EPANET closes, where the
    experiment is not EPANET: a junction a leak enters has a negative demand, which EPANET's
    quality counts as clean inflow (`qualroute.c`, `findnodequal`: `volin -= MIN(0,
    NodeDemand) tstep`), and a closed link holds what it last took from there."""
    model, _, _ = build_model(read_epanet_inp(path))
    leaks, closed = _epanets_closed_link_leaks(path, model, trials=trials)
    times = sorted(leaks)
    plain = simulate._Patterns.drivers_at

    def drivers_at(self, t):
        d = plain(self, t)
        at = max(s for s in times if s <= t)
        return {**d, "water.sources": d["water.sources"] + leaks[at]}

    with monkeypatch.context() as patch:
        patch.setattr(simulate._Patterns, "drivers_at", drivers_at)
        _, diagnostics = run_quality(path)
    injected = {model.node_names[i] for s in leaks.values()
                for i in s.nonzero().flatten().tolist()}
    return diagnostics, injected, closed


def _beyond(diagnostics: dict, reference: QualityReference, *, ctol: float,
            exclude=frozenset()) -> list[tuple[int, str]]:
    """The `(time, name)` points of noodl's quality beyond 1e-5 of the scale (`max(|EPANET|,
    ctol)`, as `assert_matches_epanet`) but for the names in `exclude`."""
    mine, theirs = _matrix(diagnostics, reference)
    names = diagnostics["node_names"] + diagnostics["link_names"]
    error = (mine - theirs).abs()
    bound = 1e-5 * max(float(theirs.abs().max()), ctol)
    return [(reference.times[i], names[j]) for i, j in (error > bound).nonzero().tolist()
            if names[j] not in exclude]


def _with_accuracy(accuracy: float):
    """An `epanet_quality` `setup` setting EPANET's ACCURACY to `accuracy` (`_open` sets
    `REFERENCE_ACCURACY`, 1e-7)."""
    def setup(en):
        code = en.ENlib.EN_setoption(en._project, EN_ACCURACY, ctypes.c_double(accuracy))
        assert code == 0, f"EN_setoption(EN_ACCURACY, {accuracy}) returned {code}"
    return setup


def _ky4_trace(tmp_path: Path) -> Path:
    """ky4 for 24 h (`test_water_networks._ky_24h`): TRACE from its reservoir R-1 and a
    QUALITY TIMESTEP (1:00) equal to its hydraulic step (1:00), both as shipped, so the file
    needs no other edit."""
    path = _ky_24h(tmp_path, "ky4")
    net = read_epanet_inp(path)
    assert (net.quality.mode, net.quality.trace_node) == ("TRACE", "R-1")
    assert net.quality.quality_timestep == net.hydraulic_timestep == 3600.0
    assert net.duration == 86400.0
    return path


#: ky4's points beyond 1e-5 of the scale (1e-3 %), each with its cause, checked in
#: `test_ky4_trace_misses_are_epanets_closed_link_leaks_and_its_accuracy`. EPANET's closed
#: links conduct `1 / CBIG`, noodl's carry nothing (Ruling 30; `test_water_networks.
#: _NETWORKS["Net6"]`): the constant-power pump ~@Pump-1 (R-1 -P-977- I-Pump-1 -pump-
#: O-Pump-1 -P-368- J-274), closed in [STATUS] to 5501 s and by T-3's level controls from
#: 23498 to 57698 s and from 83882 s, carries 8.9e-8 to 9.7e-8 m3/s back to R-1 across its 96 to
#: 105 m (`(H1 - H2) / CBIG`), and T-1's and T-2's pipes P-539, P-36 and P-541, closed at
#: their tanks' limits from 16813 and 18555 s, 4e-9 to 2e-8 m3/s each (MEASURED from
#: EPANET's heads). P-977 itself carries the pump's leak, -9e-8 m3/s, below `Q_STAGNANT`:
#: EPANET routes a stagnant link from its N1 (`transport`: `FlowDir` 0 is not `< 0`), so
#: R-1's 100 % enters P-977 at 9e-8 m3/s, where noodl's P-977 carries 0.
_KY4_P977_TAKES_THE_LEAK = ((3600, "P-977"), (5501, "P-977"))
#: The leaks' flows through the network (the hydraulic difference: up to 9.2e-8 m3/s at
#: P-368 while the pump is closed, 2e-8 to 4e-8 m3/s at P-538, P-540 and P-1150 while only
#: the tank pipes are; MEASURED), on the slow pipes downstream of the pump station.
_KY4_CLOSED_LINK_LEAKS = (
    (32400, "P-433"), (36000, "P-433"), (39600, "J-345"), (39600, "P-384"),
    (39600, "P-433"), (43200, "J-421"), (43200, "P-384"), (43200, "P-430"),
    (46800, "P-271"), (46800, "P-384"), (50400, "J-345"), (50400, "P-271"),
    (50400, "P-384"), (54000, "J-345"), (54000, "J-426"), (54000, "P-271"),
    (54000, "P-361"), (54000, "P-384"), (57600, "J-455"), (57600, "P-271"),
    (57600, "P-361"), (57600, "P-384"), (57698, "J-345"), (57698, "P-271"),
    (57698, "P-361"), (57698, "P-384"), (83882, "J-73"), (86400, "J-73"))
#: EPANET's own convergence at ACCURACY 1e-7: R-1's front reaches J-703 between 82800 and
#: 83882 s through the slow P-709 (1e-5 m3/s), whose EPANET flow moves by up to 3e-4
#: relative between hours where noodl's does not (4.7454e-6 and 4.7482e-6 m3/s at 7200 and
#: 10800 s, noodl's 4.7469e-6 at both; MEASURED); at ACCURACY 1e-8 these are gone.
_KY4_EPANETS_ACCURACY = ((83882, "J-703"), (83882, "P-696"), (86400, "P-696"))
_KY4_ADMITTED = _KY4_P977_TAKES_THE_LEAK + _KY4_CLOSED_LINK_LEAKS + _KY4_EPANETS_ACCURACY


@pytest.mark.slow
def test_ky4_trace_matches_epanet(tmp_path):
    """ky4 for 24 h (`_ky4_trace`; its hydraulics within the default bounds,
    `test_water_networks.test_ky_extended_period_hydraulics_match_epanet`): TRACE from R-1,
    every one of EPANET's 31 quality times (the step plans equal), within the spec's bounds
    but at the 33 points `_KY4_ADMITTED` names with their causes, the worst within `Ctol`
    (0.01 %).

    MEASURED: worst 9.35e-3 % (P-977 at 5501 s, 0 in noodl; scale 100 %), 99.9th percentile
    5.5e-4 %, 33 of 65782 points beyond 1e-5 of the scale (1e-3 %), the worst of the others
    5.5e-3 % (P-430 at 43200 s). The segment counts are those of the router on EPANET's
    hydraulics (`_router_on_epanets_hydraulics`) but at P-977 (two steps) and P-368 (three),
    the leak's links: no merge flips at a `Ctol` tie. Mass balance: terms within 1.9e-6
    relative, the ratio 8.97e-7 from EPANET's (0.99999997 against 0.99999908)."""
    path = _ky4_trace(tmp_path)
    results, diagnostics = run_quality(path)
    reference = epanet_quality(path)
    assert len(reference.times) == 31
    ctol = read_epanet_inp(path).quality.tolerance
    found = assert_matches_epanet(diagnostics, reference, ctol=ctol, admitted=_KY4_ADMITTED)
    assert sorted(_beyond(diagnostics, reference, ctol=ctol)) == sorted(_KY4_ADMITTED)
    assert found["worst"] < ctol
    _assert_mass_balance(path, results, reference)


@pytest.mark.slow
def test_ky4_trace_misses_are_epanets_closed_link_leaks_and_its_accuracy(tmp_path,
                                                                        monkeypatch):
    """The causes of `_KY4_ADMITTED`, checked. (1) noodl's router on EPANET's own saved
    hydraulics (`_router_on_epanets_hydraulics`) is EPANET within 1e-9 of the scale at every
    point (MEASURED 2.0e-9 %): the misses are in the flows noodl's hydraulics give it.
    (2) noodl with EPANET's closed-link leaks added as junction sources
    (`_with_epanets_closed_link_leaks`) is within 1e-5 of the scale at every point against
    EPANET at ACCURACY 1e-7 but at `_KY4_EPANETS_ACCURACY`, and against EPANET at 1e-8
    (TRIALS 5000) at every point, both but where the experiment injects the leaks (I-Pump-1,
    O-Pump-1 and the tank pipes' junctions) and on the closed links (MEASURED there up to
    50 %, I-Pump-1's leak entering as clean water). MEASURED, leaks injected: against 1e-7 J-703
    5.4e-3 % and P-696 1.4e-3 %; against 1e-8 none. (Without the leaks, EPANET at 1e-8 still
    gives 30 of the 33.)"""
    path = _ky4_trace(tmp_path)
    reference = epanet_quality(path)
    ctol = read_epanet_inp(path).quality.tolerance
    driven = assert_matches_epanet(_router_on_epanets_hydraulics(path), reference, ctol=ctol)
    assert driven["worst"] < 1e-9 * driven["scale"]
    leaky, injected, closed = _with_epanets_closed_link_leaks(path, monkeypatch)
    assert injected == {"I-Pump-1", "O-Pump-1", "J-475", "J-59f", "J-637"}
    assert closed == {"~@Pump-1", "P-36", "P-539", "P-541"}
    artifacts = injected | closed
    assert not artifacts & {name for _, name in _KY4_ADMITTED}
    assert sorted(_beyond(leaky, reference, ctol=ctol, exclude=artifacts)) == sorted(
        _KY4_EPANETS_ACCURACY)
    tight = epanet_quality(path, setup=_with_accuracy(1e-8), trials=5000)
    assert _beyond(leaky, tight, ctol=ctol, exclude=artifacts) == []


#: ky10's hydraulics part from EPANET's at 6:54:59 (`test_water_networks._KY10_EXTENDED`,
#: `test_ky10_divergence_is_epanets_dead_headed_constant_power_pump`): there ~@Pump-8 shuts
#: on T-13, and EPANET's `pumpcoeff` caps the constant-power pump ~@Pump-10's slope at CBIG,
#: driving its flow to ~0 (CV P-75 and PRV ~@RV-5 closed) where noodl's keeps to its curve
#: (`ConstantPowerPump`'s tangent): a hydraulic difference outside the quality's (Ruling
#: Q36). Up to it the hydraulics are within the default bounds.
_KY10_DIVERGES = 24899
#: ky10's EPANET: TRIALS 5000 (`test_water_networks._KY_EPANET`; at 400 it is unbalanced at
#: 3:58:15).
_KY10_TRIALS = _KY_EPANET["ky10"]["trials"]


def _ky10_age(tmp_path: Path) -> Path:
    """ky10 (AGE, QUALITY TIMESTEP 0:05) run to 6:54:59 (`_KY10_DIVERGES`): `_ky_24h`'s file
    with its DURATION 24:00 set to 6:54:59, nothing else changed. Stopped there rather than
    run for 24 h and compared to 6:54:59: the mass balance is then the compared span's, and
    EPANET's warning 6 (negative pressures at 20:52:41) is not reached."""
    path = edited(_ky_24h(tmp_path, "ky10"), tmp_path, "ky10-to-divergence.inp",
                  replace=[(" Duration           \t24:00\n", " Duration           \t6:54:59\n")])
    net = read_epanet_inp(path)
    assert (net.quality.mode, net.quality.quality_timestep) == ("AGE", 300.0)
    assert net.duration == _KY10_DIVERGES
    return path


#: P-310 (J-184 to J-145) is the only link into a dead-end branch of four junctions
#: (`_KY10_DEAD_END`), so its flow is their demand: noodl's is, EPANET's at ACCURACY 1e-7
#: (TRIALS 5000) is 9.8e-9 to 6.2e-8 m3/s (0.1 to 0.6 %) off it at every step (MEASURED,
#: `EN_FLOW` and `EN_DEMAND`), its continuity unconverged. The age in that slow pipe (1.1e-5
#: to 2.8e-5 m3/s) follows its flow.
_KY10_P310 = ((12568, "P-310"), (14285, "P-310"), (14295, "P-310"), (14400, "P-310"),
              (18000, "P-310"), (19737, "P-310"), (21600, "P-310"), (23050, "P-310"),
              (24899, "P-310"))
_KY10_DEAD_END = ("J-184", "J-185", "J-649", "J-650")
#: A closed link's leak (`_KY4_CLOSED_LINK_LEAKS`'s cause; Ruling 30).
_KY10_CLOSED_LINK_LEAKS = ((23050, "J-433"),)
_KY10_ADMITTED = _KY10_P310 + _KY10_CLOSED_LINK_LEAKS


@pytest.mark.slow
def test_ky10_age_matches_epanet(tmp_path):
    """ky10's water age (`_ky10_age`) to 6:54:59, where its hydraulics leave EPANET's
    (`_KY10_DIVERGES`; after it the quality cannot match and is not compared): every one of
    EPANET's 18 quality times (the step plans equal; the run goes on past its last report
    time, 6:00, to 6:24:10 and 6:54:59 as EPANET's does), within the spec's bounds but at the
    10 points `_KY10_ADMITTED` names with their causes, the worst within `Ctol` (0.01 h).

    MEASURED: worst 2.15e-3 h (P-310 at 24899 s; scale 6.92 h), 99.9th percentile 1.1e-5 h,
    10 of 35928 points beyond 1e-5 of the scale (6.9e-5 h); mass-balance terms within 3.2e-6
    relative, the ratio 4.8e-9 from EPANET's (1.00061733)."""
    path = _ky10_age(tmp_path)
    results, diagnostics = run_quality(path)
    reference = epanet_quality(path, trials=_KY10_TRIALS)
    assert reference.times[-3:] == [21600, 23050, _KY10_DIVERGES]
    assert len(reference.times) == 18
    ctol = read_epanet_inp(path).quality.tolerance
    found = assert_matches_epanet(diagnostics, reference, ctol=ctol, admitted=_KY10_ADMITTED)
    assert sorted(_beyond(diagnostics, reference, ctol=ctol)) == sorted(_KY10_ADMITTED)
    assert found["worst"] < ctol
    _assert_mass_balance(path, results, reference)


@pytest.mark.slow
def test_ky10_age_misses_are_epanets_continuity_and_a_closed_link_leak(tmp_path, monkeypatch):
    """The causes of `_KY10_ADMITTED`, checked. (1) noodl's router on EPANET's saved
    hydraulics is EPANET within 1e-9 of the scale (MEASURED 1.4e-12 h). (2) P-310 is the only
    link on the junctions `_KY10_DEAD_END` but for theirs among themselves: noodl's flow in it
    is their demand to rounding at every report time; EPANET's (`EN_FLOW`, `EN_DEMAND`) is
    9.8e-9 to 6.2e-8 m3/s off theirs at every one of its 18 steps (MEASURED). (3) With
    EPANET's closed-link leaks added (`_with_epanets_closed_link_leaks`) only P-310's points
    are beyond, but where the experiment injects the leaks, on the closed links and on the
    links from the junctions it injects at (MEASURED there: P-863 from I-Pump-13 and P-1050
    from O-RV-1, both stagnant, which EPANET routes from their N1, the injected junction)."""
    path = _ky10_age(tmp_path)
    reference = epanet_quality(path, trials=_KY10_TRIALS)
    ctol = read_epanet_inp(path).quality.tolerance
    driven = assert_matches_epanet(_router_on_epanets_hydraulics(path, trials=_KY10_TRIALS),
                                   reference, ctol=ctol)
    assert driven["worst"] < 1e-9 * driven["scale"]
    # (2) P-310's flow against its dead end's demand
    net = read_epanet_inp(path)
    model, _, _ = build_model(net)
    ends = {link.name: {link.u, link.v} for link in (*net.pipes, *net.pumps, *net.valves)}
    branch = set(_KY10_DEAD_END)
    into = {name for name, e in ends.items() if e & branch and not e <= branch}
    assert into == {"P-310"} and ends["P-310"] == {"J-184", "J-145"}
    assert branch <= {j.name for j in net.junctions}
    run = water_run(model, quality=False)
    link = model.link_names.index("P-310")
    demand = run.demand[:, [[j.name for j in net.junctions].index(n) for n in _KY10_DEAD_END]]
    torch.testing.assert_close(-run.flow[:, link], demand.sum(dim=1), rtol=1e-12, atol=0.0)
    gaps = []
    with epanet_session(path, trials=_KY10_TRIALS) as en:
        units = flow_units(en)
        nodes, links = node_and_link_ids(en)
        k = links.index("P-310") + 1
        members = [nodes.index(n) + 1 for n in _KY10_DEAD_END]
        en.ENopenH()
        en.ENinitH(0)
        while True:
            en.ENrunH()
            gap = en.ENgetlinkvalue(k, _EN_FLOW) + sum(en.ENgetnodevalue(j, _EN_DEMAND)
                                                       for j in members)
            gaps.append(abs(gap) * FLOW_UNITS[units])
            if en.ENnextH() <= 0:
                break
        en.ENcloseH()
    assert len(gaps) == 18 and min(gaps) > 5e-9 and max(gaps) < 1e-7, gaps
    # (3) the closed-link leaks
    leaky, injected, closed = _with_epanets_closed_link_leaks(path, monkeypatch,
                                                              trials=_KY10_TRIALS)
    on = {name for name, e in ends.items() if e & injected}
    artifacts = injected | closed | on
    assert not artifacts & {name for _, name in _KY10_ADMITTED}
    assert sorted(_beyond(leaky, reference, ctol=ctol, exclude=artifacts)) == sorted(_KY10_P310)
