"""EPANET's example networks run whole through `water_run`, and every network `wntr` ships
read by `read_epanet_inp`.

`test_network_matches_epanet` runs Net3 (168 h), Net6 (96 h), ky4 and ky10 (DURATION 0, one
steady solve) and the edited Anytown (24 h, `_anytown`, Ruling 21) for their full DURATION
against EPANET 2.2 (`wntr` 1.5.0): heads, tank levels, flows and link states at every report
time, and the pump energy section; `test_ky_extended_period_hydraulics_match_epanet` runs
ky4 and ky10 for 24 h (the quality's period), where ky10 misses from 6:54:59 for a cause
`test_ky10_divergence_is_epanets_dead_headed_constant_power_pump` shows. The bounds are
the whole-network ones of the plan: heads 1e-5 relative (to `max(|H|, 1 m)`), tank levels
2e-4 m; flows 1e-6 m3/s; states equal; the energy 1e-5 relative. A network that misses one
carries its own bound with the measured cause in `_NETWORKS`.

EPANET runs through its toolkit on the file itself (`epanet_results`) at a true
`ACCURACY 1e-7` (TRIALS 400) on every network, not at the files' 1e-3 or 1e-4: EPANET stops
when the flow change is below that fraction of the total flow (`hydsolver.c`, `newflows`),
and at the files' values its own answer is off by more than the bounds (MEASURED: Net6 at
t = 0, heads 5.7e-5 relative at 1e-3 and 3.2e-7 at 1e-5 and 1e-7; Net3, flows 3.0e-5 m3/s
at 1e-3 at 1 h; ky4, 1.1e-5 m3/s, `test_ky4_at_time_zero`). (Before, wntr's
`EpanetSimulator` ran with `ACCURACY 1e-8` in its rewrite of the file, which EPANET's reader
clamps to 1e-5, `input3.c`, `optionvalue`.)
"""

import math
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model
from noodl.apps.water.simulate import water_run

wntr = pytest.importorskip("wntr")

from tests.verification._epanet_quality import epanet_results  # noqa: E402
from tests.verification.test_water_parity import (  # noqa: E402
    _KY10_DAMPLIMIT,
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
    # closed pipe, 3-point pumps. MEASURED: heads 1.2e-7, levels 5.2e-6 m, flows 1.3e-7 m3/s
    # (3.0e-7 with EPANET at 1e-5), states equal, energy 4.2e-8. (With the exact GPM factor
    # in Hazen-Williams, before the law took EPANET's rounded one, tank 1 drifted 6.4e-6 m
    # above EPANET's by 21 h, which rounded a level trigger to a second later: levels
    # 1.3e-4 m, flows 2.1e-6 m3/s.)
    "Net3": dict(path=_library("Net3")),
    # Net6: 96 h, 3323 junctions, 61 pumps (one constant-power), two PRVs, a check valve,
    # 124 IF NODE controls, 18 pumps closed in [STATUS]. MEASURED: heads 1.10e-5 (identical
    # with EPANET at 1e-6 and 1e-7; every hour before 59 h within 5e-6), states equal at
    # every hour, levels 1.94e-3 m, flows 3.05e-4 m3/s, energy 1.95e-4 (PUMP-3866's time on
    # line, 4.4401 against 4.4410 percent: 3 s of 96 h). HEADS 2e-5 (Ruling Q15): the worst
    # is at 90 h, JUNCTION-2828 1.47e-3 m apart, following TANK-3342's level, 1.47e-3 m
    # apart then: the level drift below, which at heads of 130 m is 1.1e-5 relative; at
    # EPANET's 1e-5 (9.4e-6, its own heads there 2.4e-6 from its 1e-7 ones, no status
    # different) the default 1e-5 was met by chance. LEVELS 3e-3 m, FLOWS 5e-4 m3/s and
    # ENERGY 3e-4, all from the switching times: EPANET's closed links conduct
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
    "Net6": dict(path=_library("Net6"), heads=2e-5, levels=3e-3, flows=5e-4, energy=3e-4),
    # ky4: DURATION 0, 959 junctions, two constant-power pumps (one closed), T-2 empty.
    # MEASURED: heads 8.2e-8, levels 6.0e-6 m, flows 9.1e-8 m3/s, and 2.6e-7 on its two
    # pairs of parallel pipes (held to 3e-6 when EPANET ran at 1e-5, whose split left their
    # losses 1.0e-7 m apart: `test_ky4_at_time_zero`), states equal, energy 1.1e-8.
    "ky4": dict(path=_library("ky4")),
    # ky10: DURATION 0, thirteen constant-power pumps, PRVs, thirteen tanks, level controls
    # on `~@`-named pumps. EPANET at DAMPLIMIT 1e-2 (`test_water_parity._KY10_DAMPLIMIT`,
    # the MEASURED cause there): at 0 wntr's Linux build stalls ~@Pump-11 at zero flow,
    # ~@RV-4 CLOSED. MEASURED, both of wntr's builds: heads 1.2e-7, levels 1.3e-5 m, flows
    # 5.5e-8 m3/s, states equal, energy 9.0e-7 (Windows; Linux 4.2e-7); on Windows at
    # DAMPLIMIT 0 energy 1.1e-7 (heads 6.3e-7, flows 9.1e-8, energy 8.3e-7 at 1e-5).
    "ky10": dict(path=_library("ky10"), damplimit=_KY10_DAMPLIMIT),
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
    # 3.8e-7 partly offset that: 1.16e-5.) EPANET at 1e-5 gave the same numbers. Over the
    # day some junctions' pressures are negative, EPANET's warning 6.
    "Anytown": dict(path=_anytown_edited, heads=1e-4, warnings=(6,)),
}


def _epanet(path, warnings=(), trials=None, damplimit=None):
    """EPANET at ACCURACY 1e-7, TRIALS 400 or `trials`, DAMPLIMIT 0 or `damplimit`
    (`epanet_results`, permitting `warnings`): `(results, energy section, demand charge)`."""
    reader = _EnergyBin()
    results = epanet_results(path, reader=reader, warnings=warnings, trials=trials,
                             damplimit=damplimit)
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


def _hydraulic_differences(path, results, *, duration=None, skip_nodes=(),
                           diagnostics=None):
    """noodl (`water_run(quality=False)` on `path`, for `duration` s or the file's) against
    EPANET's `results` at every report time noodl gives: `(model, run, differences)`,
    `differences` the worst head (relative to `max(|H|, 1 m)`, the nodes `skip_nodes` left
    out), tank level (m) and flow (m3/s), each `(value, time s, element)`, and the
    `(time, link)` pairs whose states differ; `diagnostics` goes to `water_run`."""
    net = read_epanet_inp(path)
    model, _, _ = build_model(net)
    run = water_run(model, quality=False, duration=duration, diagnostics=diagnostics)
    names, links = model.node_names, model.link_names
    rows = len(run.time)
    assert run.time.tolist() == [float(t) for t in results.node["head"].index[:rows]]

    def worst(diff, labels):
        k = int(diff.argmax())
        t, j = divmod(k, diff.shape[1])
        return float(diff.flatten()[k]), float(run.time[t]), labels[j]

    heads = torch.tensor(results.node["head"][names].to_numpy()[:rows], dtype=F64)
    relative = (run.head - heads).abs() / heads.abs().clamp(min=1.0)
    relative[:, [names.index(n) for n in skip_nodes]] = 0.0
    differences = {"heads": worst(relative, names)}
    if net.tanks:
        level = torch.stack([heads[:, names.index(t.name)] - t.elevation for t in net.tanks],
                            dim=1)
        differences["levels"] = worst((run.tank_level - level).abs(),
                                      [t.name for t in net.tanks])
    else:
        differences["levels"] = (0.0, 0.0, None)
    flows = torch.tensor(results.link["flowrate"][links].to_numpy()[:rows], dtype=F64)
    differences["flows"] = worst((run.flow - flows).abs(), links)
    status = torch.tensor(results.link["status"][links].to_numpy()[:rows]).to(run.status.dtype)
    differences["states"] = [(float(run.time[t]), links[j])
                             for t, j in (run.status != status).nonzero()]
    assert not math.isnan(sum(differences[q][0] for q in ("heads", "levels", "flows")))
    return model, run, differences


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_NETWORKS))
def test_network_matches_epanet(tmp_path, name):
    """The whole network through `water_run` against EPANET at every report time, with the
    bounds of `_NETWORKS`."""
    case = _NETWORKS[name]
    path = case["path"](tmp_path)
    results, theirs, demand = _epanet(path, case.get("warnings", ()),
                                      damplimit=case.get("damplimit"))
    model, run, differences = _hydraulic_differences(path, results)
    worst_e = _worst_energy(model.water_network, run.energy, theirs, demand)
    assert not math.isnan(worst_e)
    assert not differences["states"], differences["states"]
    assert differences["heads"][0] < case.get("heads", 1e-5), differences["heads"]
    assert differences["levels"][0] < case.get("levels", 2e-4), differences["levels"]
    assert differences["flows"][0] < case.get("flows", 1e-6), differences["flows"]
    assert worst_e < case.get("energy", 1e-5), worst_e


def _ky_24h(tmp_path, name, duration="24:00"):
    """ky4 or ky10 as `wntr` ships it (DURATION 0) with its `[TIMES] Duration` set to
    `duration` (24:00 unless a test says otherwise) and nothing else changed."""
    text = _LIBRARY.joinpath(f"{name}.inp").read_text()
    line = " Duration           \t0\n"
    assert text.count(line) == 1
    tag = "24h" if duration == "24:00" else duration.replace(":", "-")
    path = tmp_path / f"{name}-{tag}.inp"
    path.write_text(text.replace(line, f" Duration           \t{duration}\n"))
    return path


#: EPANET's settings for the 24 h ky4 and ky10 (`_ky_24h`). ky10: TRIALS 5000, since at
#: TRIALS 400 EPANET (ACCURACY 1e-7) exceeds its trials at 3:29:28 and is unbalanced at
#: 3:58:15 (warnings 2 and 1), balanced at 5000 after 410 and 437 trials (MEASURED; at 1e-6
#: and 1e-5 with 400 trials those steps need 7 to 9 trials, and their heads, levels, flows
#: and states against noodl's agree with 1e-7's within 0.1 %, the same nine states
#: differing); its warning 6, negative pressures at 20:52:41 and 21:00, is the network's
#: state. DAMPLIMIT 1e-2 (`test_water_parity._KY10_DAMPLIMIT`): at 0, wntr's Linux build
#: stalls ~@Pump-11 at zero flow from time zero and its 24 h is another run (its steps from
#: 1933 s on, states at 0-4 h), MEASURED; at 1e-2 both builds take the same 50 steps, their
#: states equal and heads within 1e-6 relative at every report time but the three nodes
#: `_KY10_ISLAND` at 7-10 h, flows within 1.6e-7 m3/s, levels 3.1e-5 m.
_KY_EPANET = {"ky4": {}, "ky10": dict(trials=5000, damplimit=_KY10_DAMPLIMIT, warnings=(6,))}

#: ky10 from 6:54:59 (24899 s), the cause `test_ky10_divergence_is_epanets_dead_headed_
#: constant_power_pump` shows. MEASURED over the 24 h (EPANET as `_KY_EPANET`, both of
#: wntr's builds; at DAMPLIMIT 0, Windows, heads 5.016 at 275.9 m, the rest as here): heads
#: 4.890 relative (O-Pump-10 at 7 h, 1659.8 m against 281.8 m; 4.883 against Linux's
#: 282.1 m), levels 0.422 m (T-9, 12 h), flows 2.33e-2 m3/s (P-948, 12 h), states differing
#: at 7-10 h (P-75, ~@RV-5) and 15 h (P-5); every report time before 7 h within the bounds.
_KY10_EXTENDED = (
    "ky10 24 h: from 6:54:59 EPANET's iteration dead-heads the constant-power pump "
    "~@Pump-10 (its flow at most 1.5e-18 m3/s, CV P-75 closed) where "
    "noodl's pump runs at 5.4e-4 m3/s and 1396 m (ConstantPowerPump's tangent "
    "below EPANET's CBIG slope); MEASURED heads 4.89 relative (O-Pump-10, 7 h), "
    "levels 0.422 m (T-9, 12 h), flows 2.33e-2 m3/s (P-948, 12 h), states at 7-10 h and "
    "15 h; 0-6 h within the bounds")


@pytest.mark.slow
@pytest.mark.parametrize("name", [
    "ky4", pytest.param("ky10", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError, reason=_KY10_EXTENDED))])
def test_ky_extended_period_hydraulics_match_epanet(tmp_path, name):
    """ky4 (TRACE) and ky10 (AGE) ship with DURATION 0; their quality runs a 24 h period,
    whose hydraulics this compares with EPANET's at every report time (`_ky_24h`, EPANET as
    `_KY_EPANET`): heads 1e-5 relative, tank levels 2e-4 m, flows 1e-6 m3/s on every link,
    states equal. MEASURED, ky4: heads 1.13e-7 (J-483, 16 h), levels 2.57e-5 m (T-3, 16 h),
    flows 2.55e-7 m3/s (P-625 at 0 h; 1.9e-6 with EPANET at 1e-5), states equal. ky10
    misses from 7 h (`_KY10_EXTENDED`)."""
    path = _ky_24h(tmp_path, name)
    results, _, _ = _epanet(path, **_KY_EPANET[name])
    _, run, differences = _hydraulic_differences(path, results)
    assert run.time.tolist() == [3600.0 * k for k in range(25)]
    assert not differences["states"], differences["states"]
    assert differences["heads"][0] < 1e-5, differences
    assert differences["levels"][0] < 2e-4, differences
    assert differences["flows"][0] < 1e-6, differences


#: ky10's dead-ended branch: J-702 -P-208- I-Pump-10 -~@Pump-10 (POWER 10 hp)- O-Pump-10
#: -P-22- I-RV-5 -~@RV-5 (PRV, 150 psi)- O-RV-5 -P-75 (CV)- J-11; the nodes between the
#: pump and the check valve, which no open link joins to a reservoir or tank once all three
#: shut.
_KY10_ISLAND = ("O-Pump-10", "I-RV-5", "O-RV-5")
#: EPANET's hydraulic steps (s) from 6:54:59 to 10 h, MEASURED with the toolkit's `EN_runH`
#: (ACCURACY 1e-7, TRIALS 5000, DAMPLIMIT 1e-2); the binary output keeps only the report times.
_KY10_EPANET_STEPS = [24899, 25200, 26515, 28800, 29096, 30365, 32400, 33647, 36000]


@pytest.mark.slow
def test_ky10_divergence_is_epanets_dead_headed_constant_power_pump(tmp_path):
    """The cause of ky10's 24 h miss (`_KY10_EXTENDED`), checked. At 6:54:59 (24899 s, a
    step both take) ~@Pump-8 shuts on T-13's control and J-11's head rises past ~@RV-5's
    setting; EPANET closes the check valve P-75, which leaves the constant-power ~@Pump-10
    pumping into a dead end. EPANET's `pumpcoeff` (`hydcoeffs.c`) caps a CONST_HP pump's
    slope at CBIG, its line there `hloss = -CBIG q` a head gain that vanishes with the flow,
    and `newflows` (`hydsolver.c`) halves a CONST_HP flow whose change would turn it
    negative: the pump's flow goes to zero (the pump OPEN throughout) and it acts as a
    closed link until the valves reopen at 10:40:29 (38429 s). noodl's `ConstantPowerPump`
    continues its curve by its tangent below that slope (a law, not an iteration), so its
    pump keeps the PRV active: 5.4e-4 m3/s at 1396 m of head at 7 h, O-Pump-10 at 1659.8 m.

    EPANET as `_KY_EPANET` (DAMPLIMIT 1e-2), MEASURED with both of wntr's builds: at 7-10 h
    the pump's flow 1.5e-18 m3/s and below (Windows; Linux 4.7e-20), 0 by 10 h; ~@RV-5 OPEN
    with a reverse leak of 1.3e-8 to 1.9e-8 m3/s, so its two ends and O-Pump-10 share one
    head, set by the closed links' 1 / CBIG conductances between I-Pump-10 (263.669 m at
    7 h) and J-11 (300.755 m): 281.805 m at 7 h on Windows, 282.138 m on Linux, the head of
    a group fed only through closed links (as the fenced groups of `test_water_parity`;
    at DAMPLIMIT 0 on Windows ~@RV-5 was CLOSED, 2.3e-24 GPM in the pump at 24899 s, and the
    three heads 275.91, 275.91 and 288.33 m).

    Checked here: (1) to 6 h noodl and EPANET agree within the bounds (states equal); (2)
    at 7-10 h EPANET's pump carries no flow while OPEN (below 1e-17 m3/s), P-75 closed and
    ~@RV-5 OPEN with no more than a leak (3e-8 m3/s), the cut-off nodes at one head (to
    1e-4 m) between I-Pump-10's and J-11's, noodl's pump 5.4e-4 to 2.0e-3 m3/s at 388 to
    1396 m of head; (3) with ~@Pump-10 closed AT TIME 6:54:59 (noodl's file only) noodl
    takes EPANET's hydraulic steps from 6:54:59 to 10 h (`_KY10_EPANET_STEPS`, from the
    toolkit's `EN_runH` in a probe; the unedited run takes 26516, 30368 and 33650 s) and
    its heads, levels, flows and states to 10 h are EPANET's within the bounds (MEASURED:
    heads 4.6e-7, levels 3.5e-5 m, flows 3.4e-7 m3/s), but for the pump's own state,
    ~@RV-5's (CLOSED here, its dead end on its start, OPEN in EPANET; at zero flow between
    equal heads the state is not determined, as `test_water_parity`'s dead ends) and the
    heads of the three nodes the closed links cut off (`_KY10_ISLAND`), noodl's limit
    282.21 m for all three, the PRV's dead end tied to its inlet (MEASURED 2.2e-3 relative
    on Windows, 3.4e-3 on Linux; bound 5e-3)."""
    path = _ky_24h(tmp_path, "ky10")
    results, _, _ = _epanet(path, **_KY_EPANET["ky10"])
    _, run, differences = _hydraulic_differences(path, results, duration=6 * 3600.0)
    assert not differences["states"], differences["states"]
    assert differences["heads"][0] < 1e-5, differences
    assert differences["levels"][0] < 2e-4, differences
    assert differences["flows"][0] < 1e-6, differences

    model, run, before = _hydraulic_differences(path, results, duration=10 * 3600.0)
    assert {link for t, link in before["states"]} == {"P-75", "~@RV-5"}
    names, links = model.node_names, model.link_names
    pump = links.index("~@Pump-10")
    gain = run.head[:, names.index("O-Pump-10")] - run.head[:, names.index("I-Pump-10")]
    theirs = results.link["flowrate"]["~@Pump-10"].to_numpy()
    status = results.link["status"]["~@Pump-10"].to_numpy()
    for hour in range(7, 11):
        assert abs(theirs[hour]) < 1e-17, theirs[hour]
        assert int(results.link["status"]["P-75"].iloc[hour]) == 0, hour
        assert int(results.link["status"]["~@RV-5"].iloc[hour]) == 1, hour
        assert abs(float(results.link["flowrate"]["~@RV-5"].iloc[hour])) < 3e-8, hour
        assert int(status[hour]) == int(run.status[hour, pump])
        assert float(run.flow[hour, pump]) > 5e-4 and float(gain[hour]) > 300.0
        island = [float(results.node["head"][node].iloc[hour]) for node in _KY10_ISLAND]
        assert max(island) - min(island) < 1e-4, (hour, island)
        assert (float(results.node["head"]["I-Pump-10"].iloc[hour]) < min(island)
                and max(island) < float(results.node["head"]["J-11"].iloc[hour])), hour
        if hour == 7:  # 281.805 m (Windows), 282.138 m (Linux): one value, either build
            assert abs(island[0] - 282.0) < 0.5, island
    text = path.read_text()
    assert text.count("[CONTROLS]\n") == 1
    forced = tmp_path / "ky10-24h-pump-10-closed.inp"
    forced.write_text(text.replace(
        "[CONTROLS]\n", "[CONTROLS]\nLINK ~@Pump-10 CLOSED AT TIME 6:54:59\n"))
    diagnostics: dict = {}
    _, run, differences = _hydraulic_differences(forced, results, duration=10 * 3600.0,
                                                 skip_nodes=_KY10_ISLAND,
                                                 diagnostics=diagnostics)
    steps = [int(t) for t in diagnostics["times"] if 24899 <= t <= 36000]
    assert steps == _KY10_EPANET_STEPS, steps
    assert differences["states"] == [(3600.0 * h, link) for h in range(7, 11)
                                     for link in ("~@Pump-10", "~@RV-5")]
    assert differences["heads"][0] < 1e-5, differences
    assert differences["levels"][0] < 2e-4, differences
    assert differences["flows"][0] < 1e-6, differences
    island = [names.index(n) for n in _KY10_ISLAND]
    heads = torch.tensor(results.node["head"][names].to_numpy()[:11], dtype=F64)
    assert float(((run.head - heads)[7:, island].abs() / heads[7:, island]).max()) < 5e-3


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
    results, _, _ = _epanet(path, (6,))
    model, _, _ = build_model(read_epanet_inp(path))
    run = water_run(model, quality=False)
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
    assert worst(water_run(model2, quality=False), model2.node_names) < 1e-6


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
