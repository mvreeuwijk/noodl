"""The EPANET 2.2 water-quality reference harness (`tests/verification/_epanet_quality.py`)
and the per-feature base network `tests/data/water/quality_base.inp`.

Every quality parity test compares noodl against what this harness reads from EPANET, so its
own tests come first: the step loop reproduces EPANET's own `EN_solveQ` binary output at the
report times, the mass balance is read in full (`EN_getstatistic`) and from the report, the
bounds name the first diverging step and node, and the base network has the properties the
later per-feature cases rely on.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model, initial_state
from noodl.apps.water.quality import EXACT_LITRES_PER_M3
from noodl.apps.water.quality_react import pipe_coefficients
from noodl.apps.water.simulate import water_run
from noodl.apps.water.units import Q_STAGNANT_CFS, rounded_cfs_per_m3s

wntr = pytest.importorskip("wntr")

from tests.verification._epanet_quality import (  # noqa: E402
    QualityReference,
    _open,
    _toolkit,
    assert_matches_epanet,
    edited,
    epanet_copy,
    epanet_hydraulics,
    epanet_quality,
    epanet_results,
    first_divergence,
    mass_balance_si,
    option,
    run_quality,
)

F64 = torch.float64
DATA = Path(__file__).resolve().parents[1] / "data" / "water"
LIBRARY = Path(wntr.__file__).parent / "library" / "networks"
BASE = DATA / "quality_base.inp"

#: The file's quality unit in wntr's SI (`BinFile` converts a concentration in mg/L to kg/m3
#: and leaves a TRACE percentage as it is).
_NETWORKS = {"Net1": (DATA / "Net1.inp", 1e-3), "Net2": (LIBRARY / "Net2.inp", 1e-3),
             "Net3": (LIBRARY / "Net3.inp", 1.0)}


@pytest.mark.parametrize("name", list(_NETWORKS))
def test_the_step_harness_reproduces_epanets_binary_output(name):
    """At the report times, every node's and link's quality from the `EN_runQ`/`EN_nextQ` loop
    is EPANET's own `EN_solveQ` binary output (float32 on disk, read by wntr's `BinFile`:
    `epanet_results`, the hydraulic parity's reference) within 1e-6 relative. MEASURED:
    1.3e-7 (Net1), 1.6e-7 (Net2), 5.9e-8 (Net3); exact where the binary holds 0."""
    path, si = _NETWORKS[name]
    reference = epanet_quality(path)
    results = epanet_results(path)
    for theirs, ours, names in (
            (results.node["quality"], reference.node, reference.node_names),
            (results.link["quality"], reference.link, reference.link_names)):
        rows = [reference.times.index(int(t)) for t in theirs.index]
        binary = torch.tensor(theirs[names].to_numpy(), dtype=F64) / si
        assert len(rows) == len(theirs.index) > 1
        assert float(binary.abs().max()) > 0
        torch.testing.assert_close(ours[rows], binary, rtol=1e-6, atol=0.0)


def test_the_mass_balance_is_read_in_full_and_from_the_report():
    """Net1: the five `.rpt` terms (`writemassbalance`, `%12.5e`) and the ratio, whose `%.5f`
    in the report agrees with `EN_getstatistic(EN_MASSBALANCE)` to the printed digits."""
    reference = epanet_quality(DATA / "Net1.inp")
    assert set(reference.terms) == {"initial", "inflow", "outflow", "reacted", "final", "ratio"}
    assert abs(reference.terms["ratio"] - reference.ratio) <= 5e-6
    assert reference.terms["initial"] > 0
    assert reference.terms["reacted"] != 0      # Net1's chlorine decays
    assert 0.9 < reference.ratio < 1.1


def _net1_with_ones(tmp_path, name, replace=()):
    """Net1 with every junction's and the tank's initial quality 1 (edited, `replace` too),
    EPANET's quality on it, and the volume (m3) that quality fills: every pipe's and the
    tank's (pumps and valves start empty, a reservoir holds none)."""
    import math
    net = read_epanet_inp(DATA / "Net1.inp")
    lines = [f" {n.name}  1.0" for n in (*net.junctions, *net.tanks)]
    reference = epanet_quality(edited(DATA / "Net1.inp", tmp_path, name, replace=replace,
                                      add={"QUALITY": lines}))
    volume = sum(math.pi / 4 * p.diameter**2 * p.length for p in net.pipes)
    volume += sum(math.pi / 4 * t.diameter**2 * t.init_level for t in net.tanks)
    return reference, volume


def test_the_mass_balance_terms_convert_to_quality_times_cubic_metres(tmp_path):
    """Net1 with every node's initial quality 1 mg/L: the report's initial mass, through
    `mass_balance_si`, is the network's water volume in mg/L x m3, to the report's 6
    digits."""
    reference, volume = _net1_with_ones(tmp_path, "ones.inp")
    converted = mass_balance_si(reference.terms, "CHEMICAL")
    assert converted["initial"] == pytest.approx(volume, rel=1e-5)
    assert converted["ratio"] == reference.terms["ratio"]


def test_the_mass_balance_of_age_converts_to_hours_times_cubic_metres(tmp_path):
    """`QUALITY AGE` on Net1 with every node's initial age 1 h (`initqual` takes a node's
    initial quality for AGE, `quality.c`): the report's initial mass is age times ft3, with
    no `LperFT3` (`writemassbalance`), and through `mass_balance_si(..., "AGE")` it is the
    network's water volume in h x m3, to the report's 6 digits; read as a CHEMICAL it would
    be 28.317 times too small."""
    reference, volume = _net1_with_ones(
        tmp_path, "age.inp", replace=[(" Quality            \tChlorine mg/L",
                                       " Quality            \tAge")])
    assert bool((reference.node[0] == 1.0).all()) and float(reference.node.max()) > 1.0
    converted = mass_balance_si(reference.terms, "AGE")
    assert converted["initial"] == pytest.approx(volume, rel=1e-5)
    assert mass_balance_si(reference.terms, "CHEMICAL")["initial"] == pytest.approx(
        volume / 28.317, rel=1e-5)
    assert converted["ratio"] == reference.terms["ratio"]


def _synthetic(delta, at=7200, node="J3"):
    times = [0, 3600, 7200, 10800]
    node_names, link_names = ["J1", "J2", "J3"], ["P1", "P2"]
    theirs = torch.linspace(0.2, 1.0, 12, dtype=F64).reshape(4, 3)
    link = torch.linspace(0.1, 0.9, 8, dtype=F64).reshape(4, 2)
    reference = QualityReference(times=times, node_names=node_names, link_names=link_names,
                                 node=theirs, link=link, ratio=1.0, terms={})
    ours = theirs.clone()
    ours[times.index(at), node_names.index(node)] += delta
    diagnostics = {"node_names": node_names, "link_names": link_names,
                   "quality": [{"times": [float(t) for t in times], "node_quality": ours,
                                "link_quality": link.clone()}]}
    return diagnostics, reference


def test_the_bounds_name_the_first_divergence():
    """A point 2e-3 apart (beyond 1e-5 of the scale 1.0, inside Ctol) fails, naming its time
    and node; admitted, it passes; 1e-9 apart passes."""
    diagnostics, reference = _synthetic(2e-3)
    with pytest.raises(AssertionError, match=r"7200.*J3"):
        assert_matches_epanet(diagnostics, reference, ctol=1e-2)
    found = assert_matches_epanet(diagnostics, reference, ctol=1e-2, admitted=[(7200, "J3")])
    assert found["first"][:2] == (7200, "J3")
    assert found["worst"] == pytest.approx(2e-3, rel=1e-9)
    assert found["scale"] == 1.0
    diagnostics, reference = _synthetic(1e-9)
    found = assert_matches_epanet(diagnostics, reference, ctol=1e-2)
    assert found["first"] is None and found["worst"] <= 1e-9 * 1.01
    # beyond Ctol, even admitted
    diagnostics, reference = _synthetic(2e-2)
    with pytest.raises(AssertionError, match=r"7200.*J3"):
        assert_matches_epanet(diagnostics, reference, ctol=1e-2, admitted=[(7200, "J3")])


def test_first_divergence_is_the_earliest_step_and_its_worst_name():
    times = [0, 10, 20]
    theirs = torch.zeros(3, 3, dtype=F64)
    ours = theirs.clone()
    ours[2, 0] = 5.0          # later and larger
    ours[1, 1] = 1e-3
    ours[1, 2] = 2e-3         # the worst at the first diverging step
    assert first_divergence(times, ours, theirs, ["a", "b", "c"], limit=1e-4) == (10, "c", 2e-3)
    assert first_divergence(times, ours, theirs, ["a", "b", "c"], limit=10.0) is None


def test_the_step_reader_names_the_first_diverging_step_of_a_perturbed_copy(tmp_path):
    """Net1 with a MASS source at junction 10 switched on by a pattern at 4 h (its third
    2-hour period): the step-level comparison against the unperturbed run names the first
    step after 4 h, at junction 10."""
    base = epanet_quality(DATA / "Net1.inp")
    perturbed = epanet_quality(edited(
        DATA / "Net1.inp", tmp_path, "perturbed.inp",
        add={"SOURCES": [" 10  MASS  1000  ON4"], "PATTERNS": [" ON4  0 0 1"]}))
    assert perturbed.times == base.times
    found = first_divergence(base.times, perturbed.node, base.node, base.node_names,
                             limit=1e-9)
    assert found is not None
    time, name, _ = found
    assert time == min(t for t in base.times if t > 4 * 3600)
    assert name == "10"


def test_edited_replaces_literally_and_appends_to_sections(tmp_path):
    source = tmp_path / "src.inp"
    source.write_text("[JUNCTIONS]\n J1 0 1\n[TIMES]\n Duration 1\n[END]\n")
    out = edited(source, tmp_path, "out.inp", add={"JUNCTIONS": [" J2 0 2"],
                                                   "QUALITY": [" J1 0.5"]},
                 replace=[("Duration 1", "Duration 2")])
    assert out.read_text() == ("[JUNCTIONS]\n J1 0 1\n J2 0 2\n[TIMES]\n Duration 2\n"
                               "[QUALITY]\n J1 0.5\n[END]\n")
    with pytest.raises(AssertionError, match="Duration 9"):
        edited(source, tmp_path, "bad.inp", replace=[("Duration 9", "x")])


def test_epanets_copy_adds_only_the_accuracy_trials_and_status(tmp_path):
    """The copy is the file plus three lines before `[END]`. EPANET reads them as TRIALS 400
    and ACCURACY 1e-5: `optionvalue` (`input3.c`) clamps a file's ACCURACY to [1e-5, 1e-1],
    so the harness's `_open` sets 1e-7 through `EN_setoption` (Rulings Q11, Q14)."""
    from tests.verification._epanet_quality import EN_ACCURACY, EN_TRIALS
    copy = epanet_copy(BASE, tmp_path)
    en = _toolkit()
    en.ENopen(str(copy), str(tmp_path / "o.rpt"), str(tmp_path / "o.bin"))
    try:
        assert (option(en, EN_TRIALS), option(en, EN_ACCURACY)) == (400.0, 1e-5)
    finally:
        en.ENclose()
    en = _toolkit()
    _open(en, copy, tmp_path / "p.rpt", tmp_path / "p.bin")
    try:
        assert (option(en, EN_TRIALS), option(en, EN_ACCURACY)) == (400.0, 1e-7)
    finally:
        en.ENclose()
    original, text = BASE.read_text(), copy.read_text()
    head, _, tail = original.rpartition("[END]")
    assert text.startswith(head)
    added = text[len(head):]
    assert added.split() == ["[OPTIONS]", "ACCURACY", "1e-8", "TRIALS", "400", "[REPORT]",
                             "STATUS", "YES", "[END]"] + tail.split()
    assert copy.parent == tmp_path


def test_the_quality_base_network_has_its_properties():
    """EPANET on `quality_base.inp`: P6 (the tank's pipe) changes sign at least twice; T1 rises
    and falls inside its limits; the dead end P7 carries only EPANET's solver residual (below
    1e-8 m3/s, 1/30 of `Q_STAGNANT`, 3.2e-7 m3/s: the residual of the stopping test, not a
    flow; MEASURED up to 3.7e-9 m3/s at the harness's ACCURACY 1e-7, 1.5e-10 at 1e-8, 1.6e-8
    at the file's clamped 1e-5) at every step, every other link carries more than
    `Q_STAGNANT`; the reservoir is the 1 mg/L source and the tank sees it."""
    net = read_epanet_inp(BASE)
    hydraulics = epanet_hydraulics(BASE)
    links = hydraulics["link_names"]
    flow = hydraulics["flow"]
    stagnant = Q_STAGNANT_CFS / rounded_cfs_per_m3s("LPS")
    p6 = flow[:, links.index("P6")]
    signs = torch.sign(p6)
    assert bool((signs != 0).all())
    assert int((signs[1:] != signs[:-1]).sum()) >= 2
    tank = net.tanks[0]
    level = hydraulics["head"][:, hydraulics["node_names"].index("T1")] - tank.elevation
    assert tank.min_level < float(level.min()) < float(level.max()) < tank.max_level
    assert float(level.max()) > float(level[0]) + 1.0
    assert float((torch.cummax(level, 0).values - level).max()) > 1.0     # and falls
    assert float(flow[:, links.index("P7")].abs().max()) < 1e-8
    others = [k for k, n in enumerate(links) if n != "P7"]
    assert float(flow[:, others].abs().min()) > stagnant
    reference = epanet_quality(BASE)
    nodes = reference.node_names
    assert bool((reference.node[:, nodes.index("R1")] == 1.0).all())
    t1 = reference.node[:, nodes.index("T1")]
    assert 0.0 < float(t1[-1]) < 1.0
    assert net.quality.mode == "CHEMICAL" and net.quality.units == "mg/L"
    assert net.quality.quality_timestep == 300.0


def test_the_quality_base_hydraulics_match_epanet():
    """`water_run` on `quality_base.inp` against EPANET (its reference copy) at every hour:
    heads 1e-5 relative (to `max(|H|, 1 m)`), flows 1e-6 m3/s, the tank level 2e-4 m.
    MEASURED (EPANET at ACCURACY 1e-7): heads 1.9e-7, flows 1.2e-8 m3/s, level 1.7e-5 m."""
    net = read_epanet_inp(BASE)
    model, _, _ = build_model(net)
    run = water_run(model)
    theirs = epanet_hydraulics(BASE)
    rows = [theirs["times"].index(int(t)) for t in run.time.tolist()]
    assert len(rows) == 25
    nodes = [theirs["node_names"].index(n) for n in model.node_names]
    links = [theirs["link_names"].index(n) for n in model.link_names]
    head = theirs["head"][rows][:, nodes]
    worst_h = float(((run.head - head).abs() / head.abs().clamp(min=1.0)).max())
    worst_q = float((run.flow - theirs["flow"][rows][:, links]).abs().max())
    t1 = model.node_names.index("T1")
    worst_l = float((run.tank_level[:, 0] - (head[:, t1] - net.tanks[0].elevation)).abs().max())
    assert worst_h < 1e-5, worst_h
    assert worst_q < 1e-6, worst_q
    assert worst_l < 2e-4, worst_l


# ---- the router against EPANET (Task 3): non-reactive transport, sort and reversal ----------

def _base(tmp_path: Path) -> Path:
    return BASE


def _trace(node: str) -> Callable[[Path], Path]:
    def case(tmp_path: Path) -> Path:
        return edited(BASE, tmp_path, f"trace_{node}.inp",
                      replace=[(" Quality    CHEMICAL mg/L", f" Quality    TRACE {node}")])
    return case


def _chemical(tmp_path: Path) -> Path:
    """The base with initial qualities at junctions, the dead end and the tank."""
    return edited(BASE, tmp_path, "chemical.inp",
                  add={"QUALITY": [" J3 0.5", " J6 0.8", " J5 0.4", " T1 0.25"]})


def _dead_end(tmp_path: Path) -> Path:
    """`J5` draws 0.2 mL/s (2e-7 m3/s, below `Q_STAGNANT`, 3.15e-7 m3/s) through a `P7` of
    2 m and 25 mm (0.98 L, emptied in 82 min): `P7` is stagnant (direction 0, out of the
    sort) but moves its actual flow from `J2` to `J5`, whose quality (0.4 at the start)
    turns to `J2`'s."""
    return edited(BASE, tmp_path, "dead_end.inp",
                  replace=[(" J5   20    0", " J5   20    0.0002"),
                           (" P7   J2     J5     300     100 ",
                            " P7   J2     J5     2       25  ")],
                  add={"QUALITY": [" J5 0.4"]})


def _tank_limit(tmp_path: Path) -> Path:
    """`T1`'s maximum lowered to 7.5 m: it fills inside an hour, ending a step there."""
    return edited(BASE, tmp_path, "tank_limit.inp",
                  replace=[(" T1   80    5          0.5       10 ",
                            " T1   80    5          0.5       7.5")])


def _check_valve(tmp_path: Path) -> Path:
    return edited(BASE, tmp_path, "check_valve.inp",
                  replace=[(" P5   J3     J4     1500    150       120        0          Open",
                            " P5   J3     J4     1500    150       120        0          CV")])


def _closed_turn(tmp_path: Path) -> Path:
    """`P6` closed by time controls for the hour (8 h to 9 h) in which `T1` turns from
    filling to draining: positive, closed (direction 0), then negative."""
    return edited(BASE, tmp_path, "closed_turn.inp",
                  add={"CONTROLS": [" LINK P6 CLOSED AT TIME 8", " LINK P6 OPEN AT TIME 9"]})


def _flow_cycle(tmp_path: Path) -> Path:
    """A pump `PU2` from `J3` to `J1` drives water round `J1 -> J2 -> J3 -> J1` (and back
    through `P4`, `J1 -> J3`): a directed cycle the sort must break."""
    return edited(BASE, tmp_path, "flow_cycle.inp",
                  add={"PUMPS": [" PU2  J3     J1     HEAD C2"], "CURVES": [" C2   20    15"]})


def _past_last_whole_step(tmp_path: Path) -> Path:
    """The base with initial qualities, run for 5:30: the step from 5:00 is EPANET's whole
    hour, which ends at 6:00, past the duration."""
    return edited(BASE, tmp_path, "last_step.inp",
                  replace=[(" Duration            24:00", " Duration            5:30")],
                  add={"QUALITY": [" J3 0.5", " J6 0.8", " J5 0.4", " T1 0.25"]})


#: Every quality case by name: the `.inp` it runs, made in a temporary directory (Task 7's
#: mass balance runs them all).
QUALITY_CASES: dict[str, Callable[[Path], Path]] = {
    "base": _base, "chemical": _chemical, "trace_junction": _trace("J1"),
    "trace_reservoir": _trace("R1"), "trace_tank": _trace("T1"), "dead_end": _dead_end,
    "tank_limit": _tank_limit, "check_valve": _check_valve, "closed_turn": _closed_turn,
    "flow_cycle": _flow_cycle, "last_step": _past_last_whole_step,
}


def _residual_admitted(path: Path, diagnostics: dict, reference, *,
                       setup=None) -> list[tuple[int, str]]:
    """The points of the dead end `P7` admitted beyond 1e-5 of the scale, each with its
    cause checked. MEASURED CAUSE: EPANET's `P7` carries its solver's residual (up to
    3.7e-9 m3/s at ACCURACY 1e-7, `test_the_quality_base_network_has_its_properties`) as a
    flow, so `evalnodeoutflow` pushes `J2`'s water into it (`v = |q| dt`: a stagnant link
    moves its actual flow); noodl's `P7` carries 1e-25 m3/s and keeps its initial
    segment. Every difference at `P7` must stay within what that residual can move: the
    scale times EPANET's `sum |q| dt` through `P7` so far over `P7`'s volume. Only for a
    case whose EPANET `P7` flow stays below 1e-8 m3/s (a true residual), and none on a
    network without `P7` (Net1). `setup` is the reference's own (`epanet_quality`)."""
    if "P7" not in diagnostics["link_names"]:
        return []
    hydraulics = epanet_hydraulics(path, setup=setup)
    flow = hydraulics["flow"][:, hydraulics["link_names"].index("P7")].abs()
    if float(flow.max()) >= 1e-8:
        return []
    times = hydraulics["times"]
    steps = torch.tensor([b - a for a, b in zip(times[:-1], times[1:], strict=True)],
                         dtype=F64)
    moved = torch.cat([torch.zeros(1, dtype=F64), torch.cumsum(flow[:-1] * steps, 0)])
    pipe = next(p for p in read_epanet_inp(path).pipes if p.name == "P7")
    volume = 0.785398 * pipe.length * pipe.diameter**2
    ours = diagnostics["quality"][0]
    our_times = [round(x) for x in ours["times"]]
    column = diagnostics["link_names"].index("P7")
    theirs = reference.link[:, reference.link_names.index("P7")]
    scale = max(float(reference.node.abs().max()), float(reference.link.abs().max()))
    admitted = []
    for row, t in enumerate(reference.times):
        mine = ours["link_quality"][our_times.index(t)][column]
        bound = scale * float(moved[times.index(t)]) / volume
        difference = abs(mine - float(theirs[row]))
        assert difference <= bound + 1e-12 * scale, (t, difference, bound)
        admitted.append((t, "P7"))
    return admitted


def _compare(tmp_path: Path, case: str):
    """`case` (`QUALITY_CASES`) run by noodl and EPANET and compared by the spec's bounds
    (`assert_matches_epanet`; `P7`'s residual admitted with its cause checked):
    `(results, diagnostics, reference, found)`."""
    path = QUALITY_CASES[case](tmp_path)
    results, diagnostics = run_quality(path)
    reference = epanet_quality(path)
    ctol = read_epanet_inp(path).quality.tolerance
    found = assert_matches_epanet(diagnostics, reference, ctol=ctol,
                                  admitted=_residual_admitted(path, diagnostics, reference))
    return results, diagnostics, reference, found


def _link_flow(results, name: str) -> torch.Tensor:
    return results.flow[:, results.link_names.index(name)]


@pytest.mark.parametrize("case, node", [("trace_junction", "J1"), ("trace_reservoir", "R1"),
                                        ("trace_tank", "T1")],
                         ids=["junction", "reservoir", "tank"])
def test_trace_matches_epanet(tmp_path, case, node):
    """TRACE at a junction, the reservoir and the tank: every node starts at 0, the trace
    node at 100 (`initqual`; a tank's segment at the file's quality, 0 here), and the trace
    node holds 100 (`findnodequal`). MEASURED: within 4.2e-5 % (the tank, scale 100)
    everywhere but `P7` (`_residual_admitted`)."""
    results, *_ = _compare(tmp_path, case)
    column = results.node_names.index(node)
    assert bool((results.node_quality[:, column] == 100.0).all())
    assert results.quality_units == "%"
    others = [k for k, n in enumerate(results.node_names) if n != node]
    assert bool((results.node_quality[0, others] == 0.0).all())
    assert float(results.node_quality[:, others].max()) > 1.0


def test_chemical_transport_matches_epanet(tmp_path):
    """A non-reactive chemical from the reservoir (1 mg/L) into a network with initial
    qualities at junctions, the dead end and the tank. MEASURED: within 4e-7 mg/L
    everywhere but `P7`."""
    results, *_ = _compare(tmp_path, "chemical")
    assert results.quality_units == "mg/L"
    t1 = results.node_quality[:, results.node_names.index("T1")]
    assert float(t1[0]) == 0.25 and 0.25 < float(t1[-1]) < 1.0


def test_a_reversing_pipe_matches_epanet(tmp_path):
    """`P6`, the tank's pipe, reverses twice (the tank fills, drains, fills): its chain is
    reversed at exactly the solves where its flow changes sign (`flowdirchanged`,
    `reversesegs`), and it matches EPANET."""
    results, diagnostics, *_ = _compare(tmp_path, "base")
    sign = torch.sign(_link_flow(results, "P6"))
    flips = [round(t) for t, a, b in zip(results.time[1:].tolist(), sign[:-1], sign[1:],
                                         strict=True) if a * b < 0]
    assert len(flips) == 2
    assert diagnostics["quality"][0]["reversals"] == [(t, "P6") for t in flips]


def test_a_pump_and_a_valve_in_the_path_match_epanet(tmp_path):
    """The pump `PU1` and the valve `V1` carry the reservoir's water: they start empty
    (`initsegs`), and what their upstream node releases into them (`evalnodeoutflow`) their
    downstream node, later in the order, takes in the same quality step
    (`evalnodeinflow`), so they hold no volume after any step; their quality is then the
    mean of their end nodes' (`avgqual`), which matches EPANET's as the front passes."""
    _, diagnostics, reference, _ = _compare(tmp_path, "base")
    segments = diagnostics["quality"][0]["segments"]
    for name in ("PU1", "V1"):
        chain = diagnostics["link_names"].index(name)
        assert all(counts[chain] == 0 for counts in segments)
        link = reference.link[:, reference.link_names.index(name)]
        assert float(link[0]) == 0.0 and float(link.max()) > 0.99     # the front passed


def test_a_dead_end_below_q_stagnant_matches_epanet(tmp_path):
    """`J5` draws 2e-7 m3/s through a small `P7` (`_dead_end`): `P7` is stagnant, so out of
    every sort, but still moves its actual flow from `J2` to `J5` (`transport`), and `J5`
    turns from 0.4 to `J2`'s quality as EPANET's does. MEASURED: within 3.3e-7 mg/L."""
    results, *_ = _compare(tmp_path, "dead_end")
    flow = _link_flow(results, "P7")
    assert bool((flow.abs() < Q_STAGNANT_CFS / rounded_cfs_per_m3s("LPS")).all())
    assert bool((flow.abs() > 1e-7).all())
    j5 = results.node_quality[:, results.node_names.index("J5")]
    j2 = results.node_quality[:, results.node_names.index("J2")]
    assert float(j5[0]) == 0.4
    assert abs(float(j5[-1]) - float(j2[-1])) < 0.02


def test_a_step_shorter_than_the_quality_step_at_a_tank_limit_matches_epanet(tmp_path):
    """`T1` fills inside an hour (`_tank_limit`): the hydraulic step ends there, at a time
    that is no multiple of the 300 s quality step, so the last quality step of that
    hydraulic step is shorter (`nextqual`: `dt = MIN(Qstep, hydstep - qtime)`)."""
    _, diagnostics, reference, _ = _compare(tmp_path, "tank_limit")
    odd = [t for t in diagnostics["quality"][0]["times"] if t % 300]
    assert odd and all(round(t) in reference.times for t in odd)


def test_a_check_valve_pipe_has_no_volume_like_epanet(tmp_path):
    """Review Focus 1: `P5` made a check-valve pipe is EPANET's `CVPIPE`, not a `PIPE`
    (`types.h`), so `initsegs` gives it no segment: its chain is empty after every quality
    step (what `J3` releases into it, `J4` takes in the same step), and the network
    matches EPANET's, not a pipe's travel time."""
    _, diagnostics, *_ = _compare(tmp_path, "check_valve")
    chain = diagnostics["link_names"].index("P5")
    assert all(counts[chain] == 0 for counts in diagnostics["quality"][0]["segments"])


def test_a_pipe_through_stagnant_to_reversed_matches_epanet(tmp_path):
    """Review Focus 2: `P6` positive to 8 h, closed by a time control from 8 h to 9 h (its
    direction 0), negative from 9 h (`_closed_turn`): `flowdirchanged` reverses a chain
    only when the new direction times the old one is negative, so `P6`'s chain is not
    reversed then (water leaves it at `J4` from the segment that led towards the tank),
    and the qualities match EPANET's."""
    results, diagnostics, *_ = _compare(tmp_path, "closed_turn")
    p6 = _link_flow(results, "P6")
    hours = results.time.tolist()
    assert float(p6[hours.index(7 * 3600.0)]) > 0
    assert abs(float(p6[hours.index(8 * 3600.0)])) < 1e-15
    assert float(p6[hours.index(9 * 3600.0)]) < 0
    reversals = diagnostics["quality"][0]["reversals"]
    assert [link for _, link in reversals] == ["P6"]      # only at the refill
    assert reversals[0][0] > 9 * 3600


def test_a_directed_flow_cycle_matches_epanet(tmp_path):
    """Review Focus 3: a pump `PU2` from `J3` to `J1` (`_flow_cycle`) drives water round
    `J1 -> J2 -> J3 -> J1`: `sortnodes` runs out of nodes without inflow and
    `selectnonstacknode` breaks the cycle, and the qualities match EPANET's."""
    results, diagnostics, *_ = _compare(tmp_path, "flow_cycle")
    for name in ("P2", "P3", "PU2"):
        assert bool((_link_flow(results, name) > 0).all())
    assert max(diagnostics["quality"][0]["cycle_breaks"]) > 0


#: `EN_NodeProperty` `EN_TANKLEVEL` (`epanet2_enums.h`, v2.2).
EN_TANKLEVEL = 8


def test_a_tank_started_below_its_file_level_matches_epanet(tmp_path):
    """`T1` started at 3.0 m (the file says 5 m) through `initial_state`'s tank levels, and
    in EPANET through `EN_setnodevalue(EN_TANKLEVEL, 3.0)` (`epanet.c`: `V0 =
    tankvolume(H0)`): the tank's initial segment holds the volume at 3.0 m, not the
    file's, and the qualities match EPANET's."""
    path = _chemical(tmp_path)
    model, _, _ = build_model(read_epanet_inp(path))
    state = initial_state(model, values={"water.tank_level": {"T1": 3.0}})
    diagnostics: dict = {}
    results = water_run(model, state, quality=True, diagnostics=diagnostics)
    diagnostics["node_names"], diagnostics["link_names"] = (list(model.node_names),
                                                            list(model.link_names))

    def start_at_3(en):
        en.ENsetnodevalue(en.ENgetnodeindex("T1"), EN_TANKLEVEL, 3.0)

    reference = epanet_quality(path, setup=start_at_3)
    assert float(results.tank_level[0, 0]) == pytest.approx(3.0, abs=1e-12)
    assert_matches_epanet(diagnostics, reference, ctol=0.01,
                          admitted=_residual_admitted(path, diagnostics, reference,
                                                      setup=start_at_3))
    # the file's 5 m would mix the tank otherwise
    file_level = epanet_quality(path)
    t1 = reference.node_names.index("T1")
    assert float((file_level.node[:, t1] - reference.node[:, t1]).abs().max()) > 1e-3


def test_the_quality_has_the_gradient_of_the_initial_tank_level(tmp_path):
    """d(`T1`'s quality at 3 h)/d(its initial level) by autograd against a central
    difference (h = 1e-3 m): the initial segment's volume comes from the state's tank
    head and keeps its graph (Ruling Q5)."""
    model, _, _ = build_model(read_epanet_inp(_chemical(tmp_path)))
    t1 = model.node_names.index("T1")
    tank = model.water_network.tanks[0]

    def quality_at(level):
        state = initial_state(model)
        phi = state["water.phi"].clone()
        phi[t1] = (tank.elevation + level) * model.head_scale
        state["water.phi"] = phi
        return water_run(model, state, duration=3 * 3600, quality=True).node_quality[-1, t1]

    level = torch.tensor(4.0, dtype=F64, requires_grad=True)
    (grad,) = torch.autograd.grad(quality_at(level), level)
    h = 1e-3
    with torch.no_grad():
        central = (quality_at(torch.tensor(4.0 + h, dtype=F64))
                   - quality_at(torch.tensor(4.0 - h, dtype=F64))) / (2 * h)
    assert float(grad) != 0.0
    assert float(grad) == pytest.approx(float(central), rel=1e-5)


# ---- reactions and water age (Task 4) ---------------------------------------------------------


def _reacting(name: str, *, bulk: float = 0.0, wall: float = 0.0, lines=(), options=(),
              replace=(), quality=(" J3 0.5", " J6 0.8", " T1 0.6")) -> Callable[[Path], Path]:
    """The base with global coefficients `bulk` (per day) and `wall`, the `[REACTIONS]`
    `lines`, the `[OPTIONS]` `options` and initial qualities at junctions and the tank."""
    def case(tmp_path: Path) -> Path:
        return edited(BASE, tmp_path, f"{name}.inp",
                      replace=[(" Global Bulk  0", f" Global Bulk  {bulk}"),
                               (" Global Wall  0", f" Global Wall  {wall}"), *replace],
                      add={"REACTIONS": list(lines), "OPTIONS": list(options),
                           "QUALITY": list(quality)})
    return case


def _net1(name: str, *, replace=(), lines=()) -> Callable[[Path], Path]:
    """The committed Net1 (GPM, chlorine, bulk -0.5, wall -1) edited by `replace` and
    `[REACTIONS]` `lines`."""
    def case(tmp_path: Path) -> Path:
        return edited(DATA / "Net1.inp", tmp_path, f"{name}.inp", replace=list(replace),
                      add={"REACTIONS": list(lines)})
    return case


_NET1_6H = (" Duration           \t24:00 ", " Duration           \t6:00 ")
_NET1_WALL0 = [(" Order Wall            \t1", " Order Wall            \t0"),
               (" Global Wall           \t-1", " Global Wall           \t-0.05"), _NET1_6H]
#: `test_water_parity._DW_NET1`: Net1 under Darcy-Weisbach, every pipe 0.5 millifeet.
_DW_NET1 = [(" Headloss           \tH-W", " Headloss           \tD-W"),
            ("100         \t0           \tOpen", "0.5         \t0           \tOpen")]

_REACTION_CASES: dict[str, Callable[[Path], Path]] = {
    "bulk_order0": _reacting("bulk_order0", bulk=-1, lines=[" Order Bulk 0"]),
    "bulk_order1": _reacting("bulk_order1", bulk=-1),
    "bulk_order2": _reacting("bulk_order2", bulk=-1, lines=[" Order Bulk 2"]),
    "bulk_michaelis_menten": _reacting("bulk_mm", bulk=-1,
                                       lines=[" Order Bulk -1", " Limiting Potential 2"]),
    "bulk_limit_growth": _reacting("bulk_growth", bulk=1, lines=[" Limiting Potential 2"]),
    "bulk_limit_decay": _reacting("bulk_decay", bulk=-1, lines=[" Limiting Potential 0.3"]),
    "wall_order0_si": _reacting("wall_order0", wall=-200, lines=[" Order Wall 0"]),
    "wall_order1": _reacting("wall_order1", wall=-0.05),
    "wall_order1_mass_transfer": _reacting("wall_mass_transfer", wall=-5),
    "wall_diffusivity_zero": _reacting("wall_diffusivity_zero", wall=-0.05,
                                       options=[" Diffusivity 0"]),
    "wall_order0_diffusivity_zero": _reacting("wall_order0_diffusivity_zero", wall=-200,
                                              lines=[" Order Wall 0"],
                                              options=[" Diffusivity 0"]),
    "wall_order0_us": _net1("wall_order0_us", replace=_NET1_WALL0),
    "roughness_hazen_williams": _reacting(
        "roughness_hw", wall=-0.2, lines=[" Roughness Correlation -12", " Wall P1 -0.02"]),
    "roughness_darcy_weisbach_us": _net1("roughness_dw_us", replace=_DW_NET1,
                                         lines=[" Roughness Correlation -1"]),
    "tank_reaction": _reacting("tank_reaction", lines=[" Tank T1 -2", " Order Tank 2"]),
    "age": _reacting("age", quality=(" J3 2", " T1 5"),
                     replace=[(" Quality    CHEMICAL mg/L", " Quality    AGE")]),
    "reactive_dead_end": _reacting(
        "reactive_dead_end", bulk=-1, quality=(" J5 0.4", " T1 0.6"),
        replace=[(" P7   J2     J5     300     100       120        0          Open",
                  " P7   J2     J5     300     100       120        0          Closed")]),
    "reactive_check_valve": _reacting(
        "reactive_check_valve", bulk=-1, wall=-0.3,
        replace=[(" P5   J3     J4     1500    150       120        0          Open",
                  " P5   J3     J4     1500    150       120        0          CV")]),
}
QUALITY_CASES.update(_REACTION_CASES)


def _matches(tmp_path: Path, case: str):
    """`_compare` with the worst difference over the scale noted (`found["relative"]`)."""
    results, diagnostics, reference, found = _compare(tmp_path, case)
    found["relative"] = found["worst"] / found["scale"]
    return results, diagnostics, reference, found


def _quality(results, name: str) -> torch.Tensor:
    return results.node_quality[:, results.node_names.index(name)]


@pytest.mark.parametrize("case", ["order0", "order1", "order2", "michaelis_menten",
                                  "limit_growth", "limit_decay"])
def test_bulk_reaction_matches_epanet(tmp_path, case):
    """`bulkrate` in every pipe and the tank (the tank at the global BULK, `adjustdata`):
    orders 0, 1, 2, Michaelis-Menten (order -1 with `Climit` 2) and a limiting potential
    for growth (`kb` +1/day to 2 mg/L) and decay (-1/day to 0.3 mg/L), explicit Euler per
    quality step (`pipereact`). MEASURED: `_matches`' relative worst, in the report."""
    results, _, reference, _ = _matches(tmp_path, f"bulk_{case}")
    t1 = _quality(results, "T1")
    if case == "limit_growth":
        assert float(t1[-1]) > 1.0          # grown past the source's 1 mg/L
    elif case == "limit_decay":
        assert 0.3 < float(t1[-1]) < 0.6
    else:
        assert float(t1[-1]) < 1.0          # below the source's 1 mg/L


@pytest.mark.parametrize("case", ["order0_si", "order1", "order1_mass_transfer",
                                  "diffusivity_zero", "order0_diffusivity_zero", "order0_us"])
def test_wall_reaction_matches_epanet(tmp_path, case):
    """`wallrate` with `piperate`'s mass transfer (the Reynolds number from each hydraulic
    step's flow, `ratecoeffs`; DIFFUSIVITY 1, chlorine's, by default): order 0 (-200
    mg/m2/day in the LPS base, mass-transfer limited where the flow is low or the
    concentration small; -0.05 mg/ft2/day in the GPM Net1, 6 h) and order 1 (-0.05
    m/day), a wall coefficient large enough that the mass transfer limits it everywhere
    (-5 m/day), and DIFFUSIVITY 0 (no limitation: order 0 `BIG`, order 1 `4 kw / d`)."""
    results, *_ = _matches(tmp_path, f"wall_{case}")
    assert float(results.node_quality[-1].min()) >= 0.0


@pytest.mark.parametrize("limited, free", [("wall_order1", "wall_diffusivity_zero"),
                                           ("wall_order0_si", "wall_order0_diffusivity_zero")],
                         ids=["order1", "order0"])
def test_the_mass_transfer_limits_the_wall_reaction(tmp_path, limited, free):
    """The same wall coefficient with and without DIFFUSIVITY: in EPANET the mass transfer
    slows it by more than the parity's bound (1e-5), so the cases matched in
    `test_wall_reaction_matches_epanet` tell the two apart. MEASURED at `T1`, 24 h: order 1
    1.0e-3, order 0 (-200 mg/m2/day) 3.1e-4 apart."""
    limited = epanet_quality(_REACTION_CASES[limited](tmp_path))
    free = epanet_quality(_REACTION_CASES[free](tmp_path))
    t1 = limited.node_names.index("T1")
    assert float(limited.node[-1, t1]) - float(free.node[-1, t1]) > 1e-4


@pytest.mark.parametrize("case", ["hazen_williams", "darcy_weisbach_us"])
def test_roughness_correlation_matches_epanet(tmp_path, case):
    """ROUGHNESS CORRELATION for the pipes without their own WALL (`adjustdata`, before
    `convertunits`): H-W `F / C` on the base (`P1` keeps its own -0.02 m/day), D-W `F /
    |ln(e/d)|` on Net1 under Darcy-Weisbach (`_DW_NET1`), `e/d` in millifeet over inches."""
    _matches(tmp_path, f"roughness_{case}")


def test_tank_reaction_matches_epanet(tmp_path):
    """`tankreact`: the tank's own coefficient (`TANK T1 -2`, L/mg/day) at the tank order
    (`ORDER TANK 2`); the pipes do not react (global BULK 0)."""
    results, *_ = _matches(tmp_path, "tank_reaction")
    assert float(_quality(results, "T1")[-1]) < 0.6


def test_water_age_matches_epanet(tmp_path):
    """QUALITY AGE: every pipe and tank segment ages `dt / 3600` h per quality step
    (`pipereact`, `tankreact`); the reservoir keeps its initial age, 1 h (`[QUALITY] R1
    1.0`)."""
    results, *_ = _matches(tmp_path, "age")
    assert results.quality_units == "hrs"
    assert bool((_quality(results, "R1") == 1.0).all())
    assert float(_quality(results, "T1")[-1]) > 1.0


def test_a_reactive_dead_end_junction_matches_epanet(tmp_path):
    """`J5` behind the closed `P7` has no inflow: with a reactive chemical it takes the mean
    of its links' adjoining segments (`noflowqual`, `findnodequal`), `P7`'s decaying
    segment, not its initial 0.4 mg/L."""
    results, *_ = _matches(tmp_path, "reactive_dead_end")
    j5 = _quality(results, "J5")
    assert float(j5[0]) == 0.4
    assert float(j5[-1]) == pytest.approx(0.4 * (1 - 300 / 86400) ** (24 * 12), rel=1e-9)


def test_a_check_valve_pipe_does_not_react_like_epanet(tmp_path):
    """Review Focus 1: `P5` made a check-valve pipe under bulk and wall decay: `reactpipes`
    reacts only `Type == PIPE`, and a CVPIPE holds no segment (`initsegs`); the network
    matches EPANET's."""
    _, diagnostics, *_ = _matches(tmp_path, "reactive_check_valve")
    chain = diagnostics["link_names"].index("P5")
    assert all(counts[chain] == 0 for counts in diagnostics["quality"][0]["segments"])


# ---- sources (Task 5) -----------------------------------------------------------------------

#: A pattern of zeros and twos (hourly, three times 8 h), the sources' `SRC`.
_SRC_PATTERN = [" SRC 1 2 2 0 0 2 1 1"] * 3


def _source(name: str, line: str, *, patterned: bool = False, quality=(),
            replace=()) -> Callable[[Path], Path]:
    """The base with the `[SOURCES]` `line` (the pattern `SRC` appended to it when
    `patterned`), the `[QUALITY]` lines `quality` and the `replace` edits."""
    def case(tmp_path: Path) -> Path:
        return edited(BASE, tmp_path, f"source_{name}.inp", replace=list(replace),
                      add={"SOURCES": [line + (" SRC" if patterned else "")],
                           "PATTERNS": _SRC_PATTERN if patterned else [],
                           "QUALITY": list(quality)})
    return case


_SOURCE_LINES = {"concen_reservoir": " R1 CONCEN 2.0", "mass": " J1 MASS 1200",
                 "setpoint": " J1 SETPOINT 1.5", "flowpaced": " J1 FLOWPACED 0.5"}
_ALL_ONE = tuple(f" {n} 1.0" for n in ("J1", "J2", "J3", "J4", "J5", "J6", "J7", "T1"))
for _kind, _line in _SOURCE_LINES.items():
    for _pattern in ("constant", "pattern"):
        QUALITY_CASES[f"source_{_kind}_{_pattern}"] = _source(
            f"{_kind}_{_pattern}", _line, patterned=_pattern == "pattern")
QUALITY_CASES["source_setpoint_below"] = _source(
    "setpoint_below", " J1 SETPOINT 0.5", quality=_ALL_ONE)
QUALITY_CASES["source_concen_junction_positive"] = _source(
    "concen_junction_positive", " J6 CONCEN 0.8")
QUALITY_CASES["source_concen_junction_negative"] = _source(
    "concen_junction_negative", " J6 CONCEN 0.8",
    replace=[(" J6   15    3", " J6   15    -3")])
QUALITY_CASES["inflow_junction_without_source"] = _source(
    "inflow_junction_without_source", " J6 CONCEN 0",
    replace=[(" J6   15    3", " J6   15    -3")])
QUALITY_CASES["setpoint_baseline"] = _source(
    "setpoint_baseline", " J1 SETPOINT 0", quality=_ALL_ONE)


for _kind, _line in (("mass", " T1 MASS 1200"), ("flowpaced", " T1 FLOWPACED 0.5"),
                     ("concen", " T1 CONCEN 0.8")):
    QUALITY_CASES[f"source_tank_{_kind}"] = _source(f"tank_{_kind}", _line)
QUALITY_CASES["source_tank_none"] = _source("tank_none", " T1 CONCEN 0")


@pytest.mark.parametrize("kind", ["mass", "flowpaced", "concen"])
def test_a_tank_source_is_added_to_its_outflow_only(tmp_path, kind):
    """A source at `T1` (MASS 1200 mg/min, FLOWPACED 0.5 mg/L, CONCEN 0.8 mg/L: plain `c`,
    a tank has no negative-demand rule): `findnodequal` adds it to the water leaving the
    tank, not to the tank's own quality. `T1` drains above `Q_STAGNANT` from 8 h and refills
    from 17 h or so (`P6`'s flow changes sign again): until the water it sent out returns,
    its node quality equals the run without the source bit for bit, while `J4` downstream
    carries the source; all match EPANET's, afterwards included."""
    results, *_ = _matches(tmp_path, f"source_tank_{kind}")
    plain, *_ = _compare(tmp_path, "source_tank_none")
    t1 = results.node_names.index("T1")
    j4 = results.node_names.index("J4")
    flow = results.flow[:, results.link_names.index("P6")]
    drain = [i for i in range(len(flow)) if float(flow[i]) < 0.0]
    assert drain and float(flow[drain].max()) < -Q_STAGNANT_CFS / rounded_cfs_per_m3s("LPS")
    refill = next(i for i in range(drain[0], len(flow)) if float(flow[i]) > 0.0)
    assert refill < len(flow) - 1
    assert torch.equal(results.node_quality[: refill + 1, t1], plain.node_quality[: refill + 1, t1])
    assert float((results.node_quality[drain, j4] - plain.node_quality[drain, j4]).abs().max()
                 ) > 0.01
    assert float((results.node_quality[-1, t1] - plain.node_quality[-1, t1]).abs()) > 1e-3


@pytest.mark.parametrize("pattern", ["constant", "pattern"])
@pytest.mark.parametrize("kind", list(_SOURCE_LINES))
def test_source_matches_epanet(tmp_path, kind, pattern):
    """A CHEMICAL source of each type (CONCEN at the reservoir `R1`, 2 mg/L in place of its
    1; MASS 1200 mg/min, SETPOINT 1.5 mg/L and FLOWPACED 0.5 mg/L at `J1`), constant and
    scaled by a pattern of zeros and twos (`sourcequal` at the step's start, `Qtime`): the
    node and link qualities match EPANET's by the spec's bounds (`_matches`; `P7`'s
    residual admitted, `_residual_admitted`)."""
    results, *_ = _matches(tmp_path, f"source_{kind}_{pattern}")
    node = "R1" if kind == "concen_reservoir" else "J1"
    assert float(_quality(results, node).max()) > 1.4      # the source reached the network


def test_a_reservoir_source_whose_pattern_goes_to_zero_keeps_its_last_value(tmp_path):
    """`findnodequal` returns the node's quality when the source adds 0: a reservoir's
    pattern `1 2 2 0 0 2 1 1` of a 2 mg/L source is 0 from 3 h to 5 h, and `R1` keeps the
    4 mg/L of the hour before; it takes 2 mg/L at 6 h."""
    results, *_ = _matches(tmp_path, "source_concen_reservoir_pattern")
    r1 = _quality(results, "R1")
    assert [float(r1[h]) for h in range(1, 8)] == [2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0]
    assert float(r1[0]) == 1.0                            # the file's, before any transport


def test_setpoint_below_the_node_quality_adds_nothing(tmp_path):
    """A SETPOINT of 0.5 mg/L at `J1` under a 1.0 mg/L supply (every node starts at 1.0):
    `max(c - quality, 0)` is 0, so every quality equals the run without the source, bit for
    bit, and EPANET's."""
    with_source, *_ = _matches(tmp_path, "source_setpoint_below")
    without, *_ = _matches(tmp_path, "setpoint_baseline")
    assert torch.equal(with_source.node_quality, without.node_quality)
    assert torch.equal(with_source.link_quality, without.link_quality)
    assert bool((with_source.node_quality == 1.0).all())


def test_concen_at_a_junction_needs_a_negative_demand(tmp_path):
    """A CONCEN source at `J6` adds nothing at a demand of +3 L/s (`J6` equals the run
    without the source), and with -3 L/s (an inflow) adds `-c demand dt / volout`: `J6`'s
    water leaves at the strength 0.8 mg/L on top of what it took in; both match EPANET's."""
    positive, *_ = _matches(tmp_path, "source_concen_junction_positive")
    plain, *_ = _compare(tmp_path, "base")
    assert torch.equal(positive.node_quality, plain.node_quality)
    negative, *_ = _matches(tmp_path, "source_concen_junction_negative")
    none, *_ = _compare(tmp_path, "inflow_junction_without_source")
    assert float((negative.node_quality - none.node_quality).abs().max()) > 0.1


def test_the_mass_balance_reader_takes_non_finite_terms():
    """EPANET's report prints a non-finite term as the C runtime does (MSVC `-1.#IND`,
    `1.#INF`; elsewhere `nan`, `inf`): read as NaN and +-inf, so a failing case reports."""
    import math

    from tests.verification._epanet_quality import mass_balance_terms
    report = ("  Initial Mass:      -1.#IND0e+000\n  Mass Inflow:        1.#INF0e+000\n"
              "  Mass Outflow:      -inf\n  Mass Reacted:       nan\n"
              "  Final Mass:         1.23450e+01\n  Mass Ratio:        -1.#QNAN\n")
    terms = mass_balance_terms(report)
    assert math.isnan(terms["initial"]) and math.isnan(terms["reacted"])
    assert math.isnan(terms["ratio"])
    assert terms["inflow"] == math.inf and terms["outflow"] == -math.inf
    assert terms["final"] == 12.345


# ---- tank mixing models (Task 6) --------------------------------------------------------------

def _mixing(name: str, line: str, *, replace=(), add=None) -> Callable[[Path], Path]:
    """The base with `T1`'s `[MIXING]` line replaced by `line`, `replace` and `add` too."""
    def case(tmp_path: Path) -> Path:
        return edited(BASE, tmp_path, f"mixing_{name}.inp",
                      replace=[(" T1   MIXED", line), *replace], add=add)
    return case


#: A small `T1` (8 m wide, filling from 1 m: 50 m3 at 2 mg/L, a segment of its own) fed by a
#: reservoir whose chemical alternates hourly between 0.2 and 1.0 mg/L (`R1`'s CONCEN source
#: and its pattern `ALT`): the tank takes in one segment per quality step (FIFO, LIFO: the
#: inflow differs by far more than `Ctol`) and gives back more than its initial volume from
#: 8 h, through several of them.
_SMALL_TANK = (" T1   80    5          0.5       10        12        0",
               " T1   80    1          0.5       10        8         0")
_ALTERNATING = {"SOURCES": [" R1 CONCEN 1.0 ALT"], "QUALITY": [" T1 2.0"],
                "PATTERNS": [" ALT 0.2 1.0 0.2 1.0 0.2 1.0 0.2 1.0"] * 3}

for _model, _line in (("mixed", " T1   MIXED"), ("2comp", " T1   2COMP 0.4"),
                      ("fifo", " T1   FIFO"), ("lifo", " T1   LIFO")):
    QUALITY_CASES[f"mixing_{_model}"] = _mixing(_model, _line)
#: `V1max` = 0.65 x `Vmax` is a level of 6.5 m: `T1` (5 m at the start, 8.9 m at the end,
#: down to 6.34 m between 8 h and 16 h) crosses it upwards, downwards and upwards again.
QUALITY_CASES["mixing_2comp_crossing"] = _mixing("2comp_crossing", " T1   2COMP 0.65")
QUALITY_CASES["mixing_fifo_segments"] = _mixing(
    "fifo_segments", " T1   FIFO", replace=[_SMALL_TANK], add=_ALTERNATING)
QUALITY_CASES["mixing_lifo_segments"] = _mixing(
    "lifo_segments", " T1   LIFO", replace=[_SMALL_TANK], add=_ALTERNATING)
#: `P6` closed from 3 h to 5 h (flow 0, `vnet == 0`) while the 2COMP stagnant zone holds
#: 1 m of water (fraction 0.4: `V1max` is 4 m, the tank is at 6 m).
QUALITY_CASES["mixing_2comp_no_flow"] = _mixing(
    "2comp_no_flow", " T1   2COMP 0.4",
    add={"CONTROLS": [" LINK P6 CLOSED AT TIME 3", " LINK P6 OPEN AT TIME 5"]})


@pytest.mark.parametrize("model", ["mixed", "2comp", "fifo", "lifo"])
def test_mixing_model_matches_epanet(tmp_path, model):
    """`T1` under each `[MIXING]` model (2COMP with fraction 0.4) in the base network, the
    reservoir's 1 mg/L filling a tank that starts at 0: node and link qualities match EPANET's
    by the spec's bounds (`_matches`; `P7`'s residual admitted, `_residual_admitted`). The
    models give different tanks (each differs from MIXED's by more than 1e-3 mg/L)."""
    results, *_ = _matches(tmp_path, f"mixing_{model}")
    mixed, *_ = _compare(tmp_path, "mixing_mixed")
    if model != "mixed":
        assert float((_quality(results, "T1") - _quality(mixed, "T1")).abs().max()) > 1e-3


def test_a_2comp_tank_crossing_v1max_matches_epanet(tmp_path):
    """`V1max` at 6.5 m of `T1` (fraction 0.65): noodl's own `T1` level crosses it upwards,
    downwards and upwards again (the mixing zone fills and spills into the stagnant one, the
    stagnant one is drawn on as the tank empties); qualities match EPANET's."""
    results, *_ = _matches(tmp_path, "mixing_2comp_crossing")
    side = torch.sign(results.tank_level[:, 0] - 6.5)
    assert bool((side != 0).all())
    assert int((side[1:] > side[:-1]).sum()) >= 2 and int((side[1:] < side[:-1]).sum()) >= 1


def _drain_points(tmp_path: Path, model: str):
    """`mixing_{model}_segments` run by noodl and EPANET and compared; the points beyond 1e-5
    of the scale admitted at `T1` and `P6` only, each within its measured cause.

    CAUSE, a FRONT SHIFT. The edge of a segment reaches `P6` in one quality step. noodl's
    hydraulics and EPANET's differ by up to 2e-8 m3/s in `P6` (the base's measured 1.2e-8 and
    the float32 read-back), so by `dV = sum |dq| dt` over the drain so far (up to 4e-4 m3), and
    the edge falls that much earlier or later: that step's quality moves by at most the jump
    (EPANET's own range of that quality within an hour either side of the point, 0.4 to 1.0
    mg/L here) times `dV` over the volume of the step's outflow, `|q| Qstep` (1.5 m3). Each
    admitted point must lie within that prediction (a smaller error is no defect).
    MEASURED, error over prediction: FIFO P6 11 h 4.4e-5 / 6.0e-4 (0.07), LIFO P6 15 h
    4.2e-5 / 2.7e-4 (0.15), LIFO T1 16 h 3.4e-5 / 1.0e-4 (0.32), with `dV` 4.7e-4 to 6.4e-4 m3
    (P6 flow differences of 2e-8 m3/s and less, summed over the drain).
    Not a flipped merge: noodl at `Tolerance` 0.01001 and 0.00999 (the other sides of every
    `fabs(seg->c - cin) < Ctol` near 0.01; checked by hand) leaves these points as they are."""
    path = QUALITY_CASES[f"mixing_{model}_segments"](tmp_path)
    results, diagnostics = run_quality(path)
    reference = epanet_quality(path)
    ours = diagnostics["quality"][0]
    names = list(diagnostics["node_names"]) + list(diagnostics["link_names"])
    rows = [round(t) for t in ours["times"]]
    scale = max(float(reference.node.abs().max()), float(reference.link.abs().max()))
    hydraulics = epanet_hydraulics(path)
    p6 = hydraulics["link_names"].index("P6")
    qstep = read_epanet_inp(path).quality.quality_timestep
    hours = [int(t) for t in results.time.tolist()]       # noodl's solves, EPANET's rows
    theirs_q = hydraulics["flow"][[hydraulics["times"].index(h) for h in hours], p6]
    ours_q = results.flow[:, results.link_names.index("P6")]
    dq = (ours_q - theirs_q).abs() * torch.tensor(
        [b - a for a, b in zip(hours, hours[1:] + [hours[-1]], strict=True)], dtype=F64)
    beyond = []
    for t in reference.times:
        mine = torch.tensor(ours["node_quality"][rows.index(t)]
                            + ours["link_quality"][rows.index(t)], dtype=F64)
        row = reference.times.index(t)
        theirs = torch.cat([
            reference.node[row][[reference.node_names.index(n)
                                 for n in diagnostics["node_names"]]],
            reference.link[row][[reference.link_names.index(n)
                                 for n in diagnostics["link_names"]]]])
        error = (mine - theirs).abs()
        for column in error.gt(1e-5 * scale).nonzero().flatten().tolist():
            beyond.append((t, names[column]))
            assert names[column] in ("T1", "P6"), (t, names[column], float(error[column]))
            moved = float(dq[:sum(1 for h in hours if h < t)].sum())
            flow = abs(float(theirs_q[max(i for i, h in enumerate(hours) if h <= t)]))
            ref = (reference.node[:, reference.node_names.index(names[column])]
                   if names[column] in reference.node_names
                   else reference.link[:, reference.link_names.index(names[column])])
            near = ref[max(0, row - 1):row + 2]        # the jump: EPANET's range within +-1 h
            jump = float(near.max() - near.min())
            predicted = jump * moved / (flow * qstep)
            assert float(error[column]) <= predicted, (
                t, names[column], float(error[column]), predicted)
    assert_matches_epanet(diagnostics, reference, ctol=0.01, admitted=beyond)
    return diagnostics, beyond


@pytest.mark.parametrize("model", ["fifo", "lifo"])
def test_fifo_and_lifo_tanks_emptying_through_several_segments_match_epanet(tmp_path, model):
    """A small `T1` (`_SMALL_TANK`) fed alternating 0.2 and 1.0 mg/L: its chain
    (`diagnostics["quality"][0]["segments"]`, the count after every quality step, the tanks'
    after the links') holds at least three segments and gives most of them up while the tank
    drains (FIFO from the front, LIFO from the back: at least five fewer between 8 h and 16 h),
    and the qualities match EPANET's by the spec's bounds; every point beyond 1e-5 of the scale
    has its measured cause (`_drain_points`: the sharp edge of a segment). MEASURED beyond
    1e-5 of the scale (2e-5): FIFO 4.4e-5 mg/L (P6, 11 h); LIFO 4.2e-5 (P6, 15 h), 3.4e-5
    (T1, 16 h)."""
    diagnostics, beyond = _drain_points(tmp_path, model)
    chain = len(diagnostics["link_names"])
    counts = [c[chain] for c in diagnostics["quality"][0]["segments"]]
    assert max(counts) >= 3
    drain = counts[8 * 12:16 * 12]                  # 8 h to 16 h: `P6` carries T1's water out
    assert drain[0] - min(drain) >= 5


def test_a_2comp_tank_without_net_flow_matches_epanet(tmp_path):
    """Time controls close `P6` from 3 h to 5 h: EPANET's `P6` flow is 0 there, `vnet == 0`,
    so `tankmix2` sets the stagnant volume to 0 (`vt == 0`) and the stagnant zone's water is
    lost to the mixing zone's accounting; noodl does the same (qualities match EPANET's) and
    the run differs from the same tank left open."""
    path = QUALITY_CASES["mixing_2comp_no_flow"](tmp_path)
    hydraulics = epanet_hydraulics(path)
    p6 = hydraulics["flow"][:, hydraulics["link_names"].index("P6")]
    closed = [i for i, t in enumerate(hydraulics["times"]) if 3 * 3600 <= t < 5 * 3600]
    assert closed and bool((p6[closed] == 0.0).all())
    results, *_ = _matches(tmp_path, "mixing_2comp_no_flow")
    # noodl's own closed `P6` carries exactly 0 (a closed link's flow is set to 0), not merely
    # below `ZERO_FLOW`
    hours = [i for i, t in enumerate(results.time.tolist()) if 3 * 3600 <= t < 5 * 3600]
    assert hours and bool((results.flow[hours, results.link_names.index("P6")] == 0.0).all())
    open_, *_ = _compare(tmp_path, "mixing_2comp")
    assert float((_quality(results, "T1") - _quality(open_, "T1")).abs().max()) > 1e-4


def _mass_balance_differences(tmp_path: Path, case: str):
    """`(ours, reference, theirs, relative, ratio_difference)` for `case`: noodl's mass balance
    (`WaterResults.mass_balance`, physical mass) and EPANET's report terms in the same
    unit (`mass_balance_si`, quality x m3, times 1000 for a CHEMICAL: EPANET's rounded
    `LperFT3` is removed, Ruling Q28); `theirs` is EPANET's terms in quality x m3, `relative`
    each term's difference over the absolute reference term (0 when both are 0), and
    `ratio_difference` the ratio's absolute difference."""
    path = QUALITY_CASES[case](tmp_path)
    results, _ = run_quality(path)      # the qualities' parity is the case's own test's
    reference = epanet_quality(path)
    mode = read_epanet_inp(path).quality.mode
    theirs = mass_balance_si(reference.terms, mode)
    unit = EXACT_LITRES_PER_M3 if mode == "CHEMICAL" else 1.0
    ours = results.mass_balance
    relative = {}
    for key in ("initial", "inflow", "outflow", "reacted", "final"):
        mine, expected = float(getattr(ours, key)), theirs[key] * unit
        relative[key] = (0.0 if mine == expected == 0.0 else
                         abs(mine - expected) / max(abs(expected), 1e-300))
    return ours, reference, theirs, relative, abs(float(ours.ratio) - reference.ratio)


@pytest.mark.parametrize("case", sorted(QUALITY_CASES))
def test_mass_balance_matches_epanet(tmp_path, case):
    """Every term of EPANET's mass balance (`updatemassbalance`, the reactions, `findstoredmass`,
    `evalmassbalance`) within 2e-5 relative and the ratio within 1e-6 (the spec's bounds), over
    every parity case. The remaining unit difference is the flow-carried volumes' Ruling Q17
    scale (1 - 5.4e-6 for LPS, in the inflow, outflow and the mass they build), which the 2e-5
    bound absorbs with margin; EPANET's rounded `LperFT3` is removed on both sides."""
    ours, reference, theirs, relative, ratio = _mass_balance_differences(tmp_path, case)
    assert ratio < 1e-6, (case, float(ours.ratio), reference.ratio)
    for key, value in relative.items():
        assert value < 2e-5, (case, key, float(getattr(ours, key)), theirs[key], value)
    # a term of 0 in EPANET is exactly 0 in noodl, and conversely
    for key in relative:
        assert (float(getattr(ours, key)) == 0.0) == (theirs[key] == 0.0), (case, key)


def test_a_2comp_tank_without_net_flow_loses_mass_in_both(tmp_path):
    """`tankmix2` with `vnet == 0` drops the stagnant zone's mass: the balance's ratio is
    below 1 in noodl and in EPANET alike (the mass left the accounting without leaving as
    outflow or reaction)."""
    ours, reference, *_ = _mass_balance_differences(tmp_path, "mixing_2comp_no_flow")
    assert float(ours.ratio) < 1.0 and reference.ratio < 1.0


def test_a_duration_past_the_last_whole_step_matches_epanet(tmp_path):
    """`DURATION 5:30` with an hourly hydraulic step: the step from 5:00 would end at 6:00
    and EPANET does not transport it (`nextqual`: `Htime <= Dur`; the hydraulic step is not
    shortened to the duration), so the quality ends at 5:00 after exactly 60 quality steps of
    300 s, none after 5:00; noodl's qualities and mass balance match EPANET's by the spec's
    bounds (`_compare`, the mass balance by `test_mass_balance_matches_epanet`'s over the
    same case)."""
    results, diagnostics, reference, _ = _compare(tmp_path, "last_step")
    entry = diagnostics["quality"][0]
    assert len(entry["segments"]) == 60
    assert max(entry["times"]) == 5 * 3600 and results.time[-1] == 5 * 3600
    assert reference.times[-1] == 5 * 3600
    *_, relative, ratio = _mass_balance_differences(tmp_path, "last_step")
    assert ratio < 1e-6 and max(relative.values()) < 2e-5


def test_a_batch_runs_each_instance_as_its_own_run(tmp_path):
    """Each instance of a batch is its own `QualityRouter` and its own run. A bulk
    coefficient batched over two values equals each single run exactly (`torch.equal`: the
    same flows, the same arithmetic); the demands scaled by 0.8 and 1.2 (`water.sources`)
    equal each single run within 1e-8 of the quality's scale over the first 23 h; two
    `diagnostics` entries. A batch shares one step plan (the shortest tank time over its
    instances, `water_run`), so the instances equal their single runs only while no instance
    has a step event the others lack. MEASURED: at 0.8 times the demands `T1` fills to its
    maximum at 23:33:37 (84817 s), a step the single 1.2 run does not have, which moves the
    batched 1.2 instance by 9.8e-5 in `T1`'s quality (and its heads by 1.5e-3 m); the base
    itself, at 1 times, has no such event, and none occurs before 23 h at either multiplier."""
    path = _reacting("batch", bulk=-0.5)(tmp_path)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    coefficient, _ = pipe_coefficients(net, net.quality)
    values = torch.stack([coefficient, coefficient * 2.0])
    diagnostics: dict = {}
    batch = water_run(model, state, {**drivers, "quality.bulk_coefficient": values},
                      diagnostics=diagnostics)
    assert len(diagnostics["quality"]) == 2
    assert batch.node_quality.shape[0] == 2 and batch.link_quality.shape[0] == 2
    assert batch.mass_balance.ratio.shape == (2,)
    for i in range(2):
        alone = water_run(model, state, {**drivers, "quality.bulk_coefficient": values[i]})
        assert torch.equal(batch.node_quality[i], alone.node_quality)
        assert torch.equal(batch.link_quality[i], alone.link_quality)
        assert torch.equal(batch.mass_balance.final[i], alone.mass_balance.final)
    assert not torch.equal(batch.node_quality[0], batch.node_quality[1])

    demands = drivers["water.sources"]
    scaled = torch.stack([demands * 0.8, demands * 1.2])
    diagnostics = {}
    batch = water_run(model, state, {**drivers, "water.sources": scaled},
                      duration=23 * 3600, diagnostics=diagnostics)
    assert len(diagnostics["quality"]) == 2
    scale = float(batch.node_quality.abs().max())
    for i in range(2):
        alone = water_run(model, state, {**drivers, "water.sources": scaled[i]},
                          duration=23 * 3600)
        assert float((batch.node_quality[i] - alone.node_quality).abs().max()) < 1e-8 * scale
        assert float((batch.link_quality[i] - alone.link_quality).abs().max()) < 1e-8 * scale
    assert float((batch.node_quality[0] - batch.node_quality[1]).abs().max()) > 1e-3 * scale
