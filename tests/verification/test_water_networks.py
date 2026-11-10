"""EPANET's example networks run whole through `water_run`, and every network `wntr` ships
read by `read_epanet_inp`.

`test_network_matches_epanet` runs Net3 (168 h), Net6 (96 h), ky4 and ky10 (DURATION 0, one
steady solve) and the edited Anytown (24 h, `_anytown`, Ruling 21) for their full DURATION
against EPANET 2.2 (`wntr` 1.5.0): heads, tank levels, flows and link states at every report
time, and the pump energy section. The bounds are the whole-network ones of the plan: heads
1e-5 relative (to `max(|H|, 1 m)`), tank levels 2e-4 m; flows 1e-6 m3/s; states equal; the
energy 1e-5 relative. A network that misses one carries its own bound with the measured cause
in `_NETWORKS`.

EPANET runs at `ACCURACY 1e-8` (TRIALS 400) on every network, not at the files' 1e-3 or
1e-4: EPANET stops when the flow change is below that fraction of the total flow
(`hydsolver.c`, `newflows`), and at the files' values its own answer is off by more than the
bounds (MEASURED: Net6 at t = 0, heads 5.7e-5 relative at 1e-3 and 3.1e-7 at 1e-8; Net3,
flows 3.0e-5 m3/s at 1e-3 at 1 h; ky4, 1.1e-5 m3/s, `test_ky4_at_time_zero`).
"""

import math
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model
from noodl.apps.water.simulate import water_run

wntr = pytest.importorskip("wntr")

from tests.verification.test_water_parity import (  # noqa: E402
    _KY4_PARALLEL,
    _M3_PER_MGAL,
    _US_FLOW_UNITS,
    _anytown,
    _EnergyBin,
)

F64 = torch.float64
_WNTR = Path(wntr.__file__).parent
_LIBRARY = _WNTR / "library" / "networks"
_TESTING = _WNTR / "tests" / "networks_for_testing"


def _library(name):
    def path(tmp_path):
        return _LIBRARY / f"{name}.inp"
    return path


def _anytown_edited(tmp_path):
    path, _ = _anytown(tmp_path, placeholders=12)
    return path


#: Per network: the file (a function of `tmp_path`) and the bounds that differ from the
#: defaults (heads 1e-5 relative, levels 2e-4 m, flows 1e-6 m3/s, energy 1e-5 relative),
#: each with its measured cause and the MEASURED worst values.
_NETWORKS = {
    # Net3: 168 h, AT TIME controls on pump 10, level controls on pump 335 and pipe 330, a
    # closed pipe, 3-point pumps. MEASURED: heads 1.2e-7, levels 5.2e-6 m, flows 3.0e-7 m3/s,
    # states equal, energy 4.2e-8. (With the exact GPM factor in Hazen-Williams, before the
    # law took EPANET's rounded one, tank 1 drifted 6.4e-6 m above EPANET's by 21 h, which
    # rounded a level trigger to a second later: levels 1.3e-4 m, flows 2.1e-6 m3/s.)
    "Net3": dict(path=_library("Net3")),
    # Net6: 96 h, 3323 junctions, 61 pumps (one constant-power), two PRVs, a check valve,
    # 124 IF NODE controls, 18 pumps closed in [STATUS]. MEASURED: heads 9.4e-6, states equal
    # at every hour, levels 1.95e-3 m, flows 2.9e-4 m3/s, energy 1.95e-4 (PUMP-3866's time
    # on line, 4.4401 against 4.4410 percent: 3 s of 96 h). LEVELS 3e-3 m, FLOWS 5e-4 m3/s
    # and ENERGY 3e-4, all from the switching times: EPANET's closed links conduct
    # `1 / CBIG` (`hydcoeffs.c`: a closed link's `P = 1 / CBIG`, CBIG = 1e8 ft per cfs),
    # noodl's carry none. MEASURED at 3600 s:
    # JUNCTION-3104, the suction of pumps 3882-3884, all three closed, is fed only by
    # LINK-3584, which carries -2.559e-7 m3/s in EPANET and 0 in noodl; noodl's head drop of
    # 91.82 m across each of the closed pumps divided by CBIG is -8.531e-8 m3/s each,
    # -2.5592e-7 m3/s together. That leak fills TANK-3355 1.7e-6 m an hour faster in
    # EPANET; such drifts move level triggers that fall near a half second. The first
    # (PUMP-3845 CLOSED on TANK-3336 at 19 ft) is 386.509 s ahead in EPANET at 18535 s, from
    # its toolkit's double-precision level and inflow (`ENgetnodevalue`), so EPANET rounds
    # it to 387 s and noodl, whose TANK-3336 is 7.1e-7 m higher at 5 h than the toolkit's
    # level, takes 386 s. By 96 h 609 steps in both, every one matched, up to a minute apart
    # late in the run. (The closed-link leak is not modelled: Ruling 30.)
    "Net6": dict(path=_library("Net6"), levels=3e-3, flows=5e-4, energy=3e-4),
    # ky4: DURATION 0, 959 junctions, two constant-power pumps (one closed), T-2 empty.
    # MEASURED: heads 8.2e-8, flows 2.0e-7 m3/s off the two parallel pairs. FLOWS 3e-6 on
    # those pairs only (`_KY4_PARALLEL`: EPANET's split leaves their losses 1e-7 m apart;
    # the energy balance is checked in `test_ky4_at_time_zero`).
    "ky4": dict(path=_library("ky4"), parallel=3e-6),
    # ky10: DURATION 0, thirteen constant-power pumps, PRVs, thirteen tanks, level controls
    # on `~@`-named pumps. MEASURED: heads 6.3e-7, flows 9.1e-8 m3/s.
    "ky10": dict(path=_library("ky10")),
    # Anytown edited (Ruling 21): 24 h at a 1 min hydraulic step, three pumps on a
    # five-point curve, the tanks (at 22.5 ft, `_anytown`) empty within the first hour,
    # the demand outrunning the three pumps from the start. MEASURED: heads 9.3e-8 at
    # t = 0 and 5.1e-5 relative over the day (junction 14, whose head is 0.554 m, 5.1e-5 m
    # apart; the worst absolute is 1.8e-4 m, junction 17 at -120.8 m, 1.9e-6 relative),
    # levels 1.8e-6 m, flows 2.9e-7 m3/s, states equal, energy 3.7e-7. HEADS 1e-4: from the
    # first hour the empty tanks' pipes 142 and 143 are closed, and EPANET's closed links
    # conduct `1 / CBIG`: at this model's 68 to 159 m across them they leak 1.4e-7 to
    # 2.7e-7 m3/s out of the tanks, and EPANET's pumps deliver 1.1e-7 to 2.6e-7 m3/s less
    # than noodl's (MEASURED at 1, 3 and 12 h), the heads at the junctions far from the
    # pumps moving by up to 1.8e-4 m; relative to the node's own head that is large only
    # where the head is near the datum. Those leaks added as demands bring the heads to
    # 1.3e-7 (`test_anytown_heads_with_epanets_tank_leaks_as_demands`). (Before
    # Hazen-Williams took EPANET's rounded GPM factor, 448.831 against 448.8311688, its
    # 3.8e-7 partly offset that: 1.16e-5.)
    "Anytown": dict(path=_anytown_edited, heads=1e-4),
}


def _epanet(path, tmp_path, tag):
    """EPANET at ACCURACY 1e-8: `(results, energy section, demand charge)`."""
    reference = wntr.network.WaterNetworkModel(str(path))
    reference.options.hydraulic.accuracy = 1e-8
    reference.options.hydraulic.trials = 400
    prefix = tmp_path / tag
    results = wntr.sim.EpanetSimulator(reference).run_sim(file_prefix=str(prefix))
    reader = _EnergyBin()
    reader.read(str(prefix) + ".bin")
    return results, reader.energy, float(reader.peak_energy[0])


def _worst_energy(net, energy, theirs, demand):
    """The worst relative difference of every pump's six energy figures and the demand
    charge (as `test_water_parity._compare_energy`)."""
    volume = _M3_PER_MGAL if net.options.flow_units in _US_FLOW_UNITS else 1.0
    assert set(theirs) == set(energy.pump_names)
    worst = 0.0
    for k, name in enumerate(energy.pump_names):
        ours = [float(energy.utilisation[k]), float(energy.efficiency[k]),
                float(energy.energy_per_volume[k]) * volume, float(energy.average_power[k]),
                float(energy.peak_power[k]), float(energy.cost[k])]
        for a, b in zip(ours, theirs[name], strict=True):
            worst = max(worst, abs(a - b) / (abs(b) or 1.0))
    return max(worst, abs(float(energy.demand_charge) - demand) / (abs(demand) or 1.0))


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_NETWORKS))
def test_network_matches_epanet(tmp_path, name):
    """The whole network through `water_run` against EPANET at every report time, with the
    bounds of `_NETWORKS`."""
    case = _NETWORKS[name]
    path = case["path"](tmp_path)
    results, theirs, demand = _epanet(path, tmp_path, name)
    net = read_epanet_inp(path)
    model, _, _ = build_model(net)
    run = water_run(model)
    names, links = model.node_names, model.link_names
    assert run.time.tolist() == [float(t) for t in results.node["head"].index]
    heads = torch.tensor(results.node["head"][names].to_numpy(), dtype=F64)
    worst_h = float(((run.head - heads).abs() / heads.abs().clamp(min=1.0)).max())
    worst_l = max((float((run.tank_level[:, k] - (heads[:, names.index(t.name)]
                                                 - t.elevation)).abs().max())
                   for k, t in enumerate(net.tanks)), default=0.0)
    flows = torch.tensor(results.link["flowrate"][links].to_numpy(), dtype=F64)
    dq = (run.flow - flows).abs()
    parallel = [links.index(p) for pair in _KY4_PARALLEL for p in pair] if "parallel" in case \
        else []
    if parallel:
        assert float(dq[:, parallel].max()) < case["parallel"]
        dq[:, parallel] = 0.0
    worst_q = float(dq.max())
    status = torch.tensor(results.link["status"][links].to_numpy()).to(run.status.dtype)
    mismatched = [(float(run.time[t]), links[j]) for t, j in (run.status != status).nonzero()]
    worst_e = _worst_energy(net, run.energy, theirs, demand)
    assert not math.isnan(worst_h + worst_l + worst_q + worst_e)
    assert not mismatched, mismatched
    assert worst_h < case.get("heads", 1e-5), worst_h
    assert worst_l < case.get("levels", 2e-4), worst_l
    assert worst_q < case.get("flows", 1e-6), worst_q
    assert worst_e < case.get("energy", 1e-5), worst_e


#: Anytown's tank pipes and the junction each joins its tank to.
_ANYTOWN_TANK_PIPES = (("142", "21", "41"), ("143", "22", "42"))
_GPM_PER_M3S = 1000.0 / 3.785411784 * 60.0
_CFS_M3 = 0.3048**3


@pytest.mark.slow
def test_anytown_heads_with_epanets_tank_leaks_as_demands(tmp_path):
    """The cause of Anytown's head bound (`_NETWORKS`), checked: the tanks start at 22.5 ft
    (`_anytown`) and empty within the first hour, after which their pipes 142 and 143 are
    closed, and EPANET's closed links conduct `1 / CBIG`. Adding to this model, as an
    inflow at junctions 21 and 22 hour by hour, the flow `dh / CBIG` (ft and cfs) its own
    heads drive through each closed tank pipe brings the heads to 1e-6 relative of EPANET's
    at every hour (MEASURED: 1.28e-7, from 5.13e-5 without). The leak enters as a junction
    of demand -1 per pipe, on an hourly pattern; the 24 h report repeats the pattern's
    first hour (tanks open, no leak) and is left out."""
    path, _ = _anytown(tmp_path, placeholders=12)
    results, _, _ = _epanet(path, tmp_path, "anytown")
    model, _, _ = build_model(read_epanet_inp(path))
    run = water_run(model)
    nodes, links = model.node_names, model.link_names
    theirs = torch.tensor(results.node["head"][nodes].to_numpy(), dtype=F64)[:24]

    def worst(run, names):
        cols = [names.index(n) for n in nodes]
        return float(((run.head[:24, cols] - theirs).abs()
                      / theirs.abs().clamp(min=1.0)).max())

    assert worst(run, nodes) > 1e-5
    text = path.read_text()
    junctions, pipes, patterns = "", "", ""
    for pipe, junction, tank in _ANYTOWN_TANK_PIPES:
        leaks = []
        for k in range(24):
            closed = int(run.status[k, links.index(pipe)]) == 0
            dh = float(run.head[k, nodes.index(tank)] - run.head[k, nodes.index(junction)])
            leak = dh / 0.3048 / 1e8 * _CFS_M3 if closed else 0.0      # m3/s into junction
            leaks.append(leak * _GPM_PER_M3S)
        assert any(leaks)
        junctions += f" L{junction}\t0\t-1\tLK{junction}\n"
        pipes += f" PL{junction}\t{junction}\tL{junction}\t1\t12\t120\t0\tOpen\n"
        patterns += f" LK{junction}\t" + "\t".join(f"{v:.10g}" for v in leaks) + "\n"
    for section, lines in (("[RESERVOIRS]", junctions), ("[PIPES]", pipes),
                           ("[PATTERNS]", patterns)):
        assert text.count(section) == 1
        if section == "[RESERVOIRS]":
            text = text.replace(section, lines + "\n" + section)
        else:
            head, rest = text.split(section)
            first, _, tail = rest.partition("\n")
            text = head + section + first + "\n" + lines + tail
    leaky = tmp_path / "anytown-leaky.inp"
    leaky.write_text(text)
    model2, _, _ = build_model(read_epanet_inp(leaky))
    assert worst(water_run(model2), model2.node_names) < 1e-6


#: The outcome of reading and building each network `wntr` ships: `None` builds, a string is
#: the refusal its `ValueError` must match (naming the feature it refuses). `bad_syntax.inp`
#: and `bad_values.inp` are malformed on purpose and not listed. At the start of the project
#: most of these were refused for features since built (pumps, valves, rules, patterns,
#: volume curves, energy); today all but one build.
_LOAD_COVERAGE = {
    "library/networks/Net1.inp": None,
    "library/networks/Net2.inp": None,
    "library/networks/Net3.inp": None,
    "library/networks/Net6.inp": None,
    "library/networks/ky4.inp": None,
    "library/networks/ky10.inp": None,
    "tests/networks_for_testing/Anytown.inp": None,
    "tests/networks_for_testing/Anytown_multipointcurves.inp": None,
    "tests/networks_for_testing/Awumah_layout1.inp": None,
    "tests/networks_for_testing/Awumah_layout8.inp": None,
    # a day count in a time ('0:00:00:00'), which EPANET does not read either
    "tests/networks_for_testing/bad_times.inp": r"is not a time of the form HH:MM\[:SS\]",
    "tests/networks_for_testing/CCWI17-HermanMahmoud.inp": None,
    "tests/networks_for_testing/conditional_controls_1.inp": None,
    "tests/networks_for_testing/conditional_controls_2.inp": None,
    "tests/networks_for_testing/control_comb.inp": None,
    "tests/networks_for_testing/cv_controls.inp": None,
    "tests/networks_for_testing/epanet_leaks.inp": None,
    "tests/networks_for_testing/fcv_open_no_downstream_sources.inp": None,
    "tests/networks_for_testing/fcv_open_no_upstream_sources.inp": None,
    "tests/networks_for_testing/io.inp": None,
    "tests/networks_for_testing/latin1.inp": None,
    "tests/networks_for_testing/leaks.inp": None,
    "tests/networks_for_testing/msx_example.inp": None,
    "tests/networks_for_testing/Net6_plus.inp": None,
    "tests/networks_for_testing/prv_closed_no_upstream_sources.inp": None,
    "tests/networks_for_testing/prv_open_no_upstream_sources.inp": None,
    "tests/networks_for_testing/psv_open_no_downstream_sources.inp": None,
    "tests/networks_for_testing/simulator.inp": None,
    "tests/networks_for_testing/skeletonize.inp": None,
    "tests/networks_for_testing/tank_controls_1.inp": None,
    "tests/networks_for_testing/tank_controls_2.inp": None,
    "tests/networks_for_testing/time_controls.inp": None,
    "tests/networks_for_testing/times.inp": None,
    "tests/networks_for_testing/Todini_Fig2_optCost_CMH.inp": None,
    "tests/networks_for_testing/Todini_Fig2_optCost_GPM.inp": None,
    "tests/networks_for_testing/Todini_Fig2_solA_CMH.inp": None,
    "tests/networks_for_testing/Todini_Fig2_solA_GPM.inp": None,
}

_MALFORMED = {"bad_syntax.inp", "bad_values.inp"}


def test_every_wntr_network_is_listed():
    """`_LOAD_COVERAGE` lists every `.inp` under `wntr`'s `library/networks` and
    `tests/networks_for_testing` but the two malformed ones."""
    found = {p.relative_to(_WNTR).as_posix()
             for d in (_LIBRARY, _TESTING) for p in d.rglob("*.inp")
             if p.name not in _MALFORMED}
    assert found == set(_LOAD_COVERAGE)


@pytest.mark.parametrize("name", list(_LOAD_COVERAGE))
def test_every_wntr_network_builds_or_names_its_refusal(name):
    """Each network builds a model, or is refused by a `ValueError` naming what it refuses
    (`_LOAD_COVERAGE`); any other exception fails."""
    path = _WNTR / name
    refusal = _LOAD_COVERAGE[name]
    if refusal is None:
        model, state, drivers = build_model(read_epanet_inp(path))
        assert len(model.node_names) > 0 and len(model.link_names) > 0
    else:
        with pytest.raises(ValueError, match=refusal):
            build_model(read_epanet_inp(path))
