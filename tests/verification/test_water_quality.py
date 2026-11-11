"""The EPANET 2.2 water-quality reference harness (`tests/verification/_epanet_quality.py`)
and the per-feature base network `tests/data/water/quality_base.inp`.

Every quality parity test compares noodl against what this harness reads from EPANET, so its
own tests come first: the step loop reproduces EPANET's own `EN_solveQ` binary output at the
report times, the mass balance is read in full (`EN_getstatistic`) and from the report, the
bounds name the first diverging step and node, and the base network has the properties the
later per-feature cases rely on.
"""

from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model
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
