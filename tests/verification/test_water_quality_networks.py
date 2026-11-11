"""EPANET's example networks' water quality run whole through `water_run` against EPANET 2.2
(`wntr` 1.5.0's toolkit, `tests/verification/_epanet_quality.py`): Net1 (the committed
`tests/data/water/Net1.inp`, chlorine, 24 h), Net2 (fluoride, a CONCEN source on a pattern
at junction 1's negative demand, 55 h), Net3 (TRACE from the Lake, 168 h) and Net6 (zero
quality as shipped; a CONCEN source at its reservoir added, 96 h), `wntr`'s files as
shipped unless a test says otherwise.

The bounds are the spec's (Verification): node and link quality over the whole (time, node
and link) matrix, every point within 1e-5 of the quality's scale (`assert_matches_epanet`,
the 99.9th percentile made strict, Ruling Q13) and the worst within `Ctol`; the mass balance
(`_mass_balance`) with every term within 2e-5 relative and the ratio within 1e-6. Each test
names its MEASURED numbers beside its bounds; a bound that differs has its measured cause in
the test. Net2, Net3 and Net6 are `slow`.
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model
from noodl.apps.water.quality import EXACT_LITRES_PER_M3, QualityRouter, quality_step
from noodl.apps.water.simulate import water_run
from noodl.apps.water.units import EPANET_QCF, rounded_cfs_per_m3s

wntr = pytest.importorskip("wntr")

from tests.verification._epanet_quality import (  # noqa: E402
    _FLOW_UNIT_CODES,
    QualityReference,
    assert_matches_epanet,
    edited,
    epanet_hydraulics,
    epanet_quality,
    epanet_results,
    epanet_session,
    first_divergence,
    mass_balance_si,
    run_quality,
)

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


def _router_on_epanets_hydraulics(path: Path) -> dict:
    """noodl's `QualityRouter` driven by EPANET's own hydraulics on `path` instead of
    noodl's: at every `EN_runH` time the quality is recorded, then the router advances over
    EPANET's step (`EN_nextH`; only while `t + step` is within the duration, `nextqual`)
    with EPANET's link flows, CLOSED states and junction demands (`NodeDemand`) as its
    quality reads them back from its hydraulics file (float32 in cfs, `savehyd`; closed
    links 0), in m3/s. The diagnostics, named as `assert_matches_epanet` takes them."""
    net = read_epanet_inp(path)
    model, _, _ = build_model(net)
    diagnostics: dict = {"node_names": list(model.node_names),
                         "link_names": list(model.link_names)}
    with epanet_session(path) as en:
        units = _FLOW_UNIT_CODES[en.ENgetflowunits()]
        qcf, cfs = EPANET_QCF[units], rounded_cfs_per_m3s(units)
        nodes = [en.ENgetnodeid(k) for k in range(1, en.ENgetcount(0) + 1)]
        links = []
        for k in range(1, en.ENgetcount(2) + 1):
            buffer = ctypes.create_string_buffer(64)
            assert en.ENlib.EN_getlinkid(en._project, k, buffer) == 0
            links.append(buffer.value.decode("utf-8"))
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


#: Net2's first merge tie: pipe 1 from junction 1 (the source) carries 0.09 mg/L through
#: hour 30 and takes 0.08 mg/L from 31 h (pattern 3), 0.01 mg/L apart: `Ctol` exactly.
_NET2_FIRST_TIE = 31 * 3600
#: Pattern 3 and EPANET's junction-1 demand (cfs) as its hydraulics file saves them
#: (float32; `savehyd`, `output.c`) at 30 h and 31 h, MEASURED (`EN_savehydfile`).
_NET2_JUNCTION_1 = ((0.09, -1.3924171924591064), (0.08, -0.6962085962295532))
#: EPANET's next tie it decides the other way (0.08 to 0.09 mg/L at 37 h, measured at 38 h).
_NET2_EPANET_MERGES = 38 * 3600


@pytest.mark.slow
def test_net2_as_shipped_differs_from_epanet_only_at_its_ctol_ties(tmp_path):
    """Net2 as shipped (`Ctol` 0.01 mg/L) matches EPANET within 1e-5 of the scale up to
    32 h and within `Ctol` everywhere; the rest is a decision at a `Ctol` tie, which the
    spec does not admit as accuracy (Ruling Q25) and this test identifies.

    MEASURED: 791 of 4256 points beyond 1e-5 of the scale, all from 32 h (115200 s), the
    first at pipe 1 (0.08283 mg/L in noodl, 0.08 in EPANET); worst 2.83e-3 mg/L there.
    CAUSE: pattern 3 steps from 0.09 to 0.08 at 31 h (and at 49 h; 0.08 to 0.09 at 37 h),
    and junction 1's outflow, `c = C0 F` (`findsourcequal`; junction 1 has no other inflow
    and pipe 1 is its only outflow, so `-D dt / volout` is 1), meets pipe 1's last segment
    at exactly `Ctol` (`evalnodeoutflow`: merged when `|c_seg - c| < Ctol`). noodl decides
    in mg/L: `0.09 - 0.08 = 0.009999999999999995 < 0.01`, merged. EPANET decides in its
    internal unit (`Ucf[QUALITY] = 1 / LperFT3`, `input1.c`, `initunits`; `C0 / Ucf`,
    `sourcequal`), where `-c D dt / volout` on its saved float32 demand (`D` and pipe 1's
    flow equal and opposite) rounds 28.317 x 0.09 up by an ulp at 30 h: `|2.5485300000000004
    - 2.2653600000000003| = 0.28317000000000014`, not below `Ctol / Ucf = 0.28317`, a new
    segment (`_NET2_JUNCTION_1`, read from `EN_savehydfile`). At 37 h EPANET
    merges (its segment of 0.08 is a few ulps off after twelve merges of equal values). The
    test checks the decisions from the runs: noodl at 0.01 is noodl at 0.01 + 1e-8 at every
    point (merged at every tie); EPANET at 0.01 is EPANET at 0.01 - 1e-8 until 38 h
    (not merged at 31 h), then neither; pipe 1 holds one segment in noodl after the first
    quality step of hour 31, two where the tie is not merged. Off the tie both agree within
    1e-5 of the scale (`test_net2_quality_matches_epanet`). Not a bug: the tie is decided by
    the last bit of the arithmetic, which noodl, running in SI with float64 flows, does not
    reproduce; noodl's router fed EPANET's own saved hydraulics makes the same decisions.
    """
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
    # the arithmetic of the two decisions: EPANET's `findsourcequal` in its internal unit
    # on its saved float32 demands (cfs), noodl's in mg/L
    ucf = 1.0 / 28.317
    epanet = [-(1.0 / ucf * f) * d * 300 / (-d * 300) for f, d in _NET2_JUNCTION_1]
    assert not abs(epanet[0] - epanet[1]) < ctol / ucf
    assert abs(0.09 - 0.08) < ctol
    # the decisions, from the runs
    below, above = (_net2_tolerance(tmp_path, t) for t in ("0.00999999", "0.01000001"))
    _, merged = run_quality(above)
    assert torch.equal(_matrix(merged, reference)[0], mine)
    split_results, split = run_quality(below)
    epanet_below = epanet_quality(below)
    early = [k for k, t in enumerate(reference.times) if t < _NET2_EPANET_MERGES]
    assert torch.equal(epanet_below.node[early], reference.node[early])
    assert torch.equal(epanet_below.link[early], reference.link[early])
    assert not torch.equal(epanet_quality(above).link, reference.link)
    link = diagnostics["link_names"].index("1")
    step = _NET2_FIRST_TIE // 300     # the quality steps of 300 s before hour 31's first
    assert diagnostics["quality"][0]["segments"][step][link] == 1
    assert split["quality"][0]["segments"][step][link] == 2
    # not the hydraulics: noodl's router on EPANET's own saved hydraulics makes the same
    # decision (MEASURED: the same 791 points, the same worst)
    driven = _router_on_epanets_hydraulics(NET2)
    on_theirs = _matrix(driven, reference)[0]
    assert first_divergence(reference.times, on_theirs, theirs, names,
                            limit=1e-5 * scale)[0] == _NET2_FIRST_TIE + 3600
    assert float((on_theirs - mine).abs().max()) < 1e-5 * scale
    # the mass balance is within the spec's bounds even so (MEASURED: outflow 1.6e-5, final
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
    within the spec's bounds for as long as noodl's hydraulic step plan is EPANET's, and the
    mass balance within them over the whole run; past that the difference is the
    hydraulics', not the quality's.

    MEASURED: before 18922 s (40 steps) worst 5.9e-6 mg/L, 99.9th percentile 7.0e-7 (scale
    1 mg/L); mass-balance terms within 1.9e-6 relative, the ratio 8.8e-8 from EPANET's.
    From 18922 s EPANET's step plan and noodl's part (`_NET6_PLANS_PART`): 252 of EPANET's
    609 step times are 1 to 55 s from noodl's, and the source's fronts reach the junctions
    at shifted times. Over the 357 times both share: worst 0.57 mg/L (JUNCTION-2292 at
    107083 s, 0.317 against 0.888), 99.9th percentile 1.1e-2, 32 % of the points beyond
    1e-5 of the scale. ADMITTED, with that measured cause: the step plan's (Net6's hydraulic
    parity: levels 3e-3 m, flows 5e-4 m3/s at its switching times, the closed-link leak noodl
    does not model, Ruling 30). The router itself is held to the spec's bounds over the
    whole run on EPANET's own hydraulics
    (`test_the_router_on_epanets_net6_hydraulics_matches_epanet`)."""
    path = _net6_with_a_source(tmp_path)
    results, diagnostics = run_quality(path)
    reference = epanet_quality(path)
    ours = [round(float(t)) for t in diagnostics["quality"][0]["times"]]
    parted = [t for t in reference.times if t not in ours]
    assert parted[0] == _NET6_PLANS_PART and _NET6_PLANS_PART - 1 in ours
    ctol = read_epanet_inp(path).quality.tolerance
    found = assert_matches_epanet(diagnostics, _until(reference, _NET6_PLANS_PART), ctol=ctol)
    assert found["worst"] < 1e-5 * found["scale"]
    shared = [k for k, t in enumerate(reference.times) if t in ours]
    common = QualityReference(times=[reference.times[k] for k in shared],
                              node_names=reference.node_names,
                              link_names=reference.link_names, node=reference.node[shared],
                              link=reference.link[shared], ratio=reference.ratio,
                              terms=reference.terms)
    mine, theirs = _matrix(diagnostics, common)
    assert bool(torch.isfinite(mine).all()) and float(mine.min()) >= 0.0
    assert float(mine.max()) <= 1.0 + 1e-12      # the source's 1 mg/L bounds every point
    _assert_mass_balance(path, results, reference)


@pytest.mark.slow
def test_the_router_on_epanets_net6_hydraulics_matches_epanet(tmp_path):
    """The cause of the admission in `test_net6_with_an_added_source_matches_epanet`, checked:
    noodl's router driven by EPANET's own saved hydraulics (`_router_on_epanets_hydraulics`)
    on Net6 with the source matches EPANET's quality at all 609 of its times, every point
    within 1e-5 of the scale, no point admitted. MEASURED: worst 5.2e-11 mg/L, 99.9th
    percentile 4.7e-11 (with EPANET's flows in double precision instead of its saved
    float32: worst 1.8e-5 at one point, JUNCTION-1361 at 236260 s, the float32 floor)."""
    path = _net6_with_a_source(tmp_path)
    driven = _router_on_epanets_hydraulics(path)
    reference = epanet_quality(path)
    assert len(reference.times) == 609
    found = assert_matches_epanet(driven, reference,
                                  ctol=read_epanet_inp(path).quality.tolerance)
    assert found["worst"] < 1e-8 * found["scale"]
