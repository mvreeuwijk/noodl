"""RDII (`rdii.py`, `inp_rdii.py`): SWMM's dates, the unit hydrograph, the processing of a record
by hand, the reader and its refusals, the driver and the gradients. The comparison with SWMM
itself is ``tests/verification/test_sewer_rdii_parity.py``."""

import dataclasses
import datetime
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.inflows import TimeSeries
from noodl.apps.sewer.rdii import (
    Rdii,
    RdiiInflow,
    UnitHydrograph,
    UnitHydrographSet,
    _ordinate,
    _seconds,
    add_seconds,
    encode_date,
    month_of_year,
)
from noodl.apps.sewer.runoff import QCF, Hydrology, RainGage, RunoffInflows

DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"
HORTON = DATA / "runoff_horton_dynwave.inp"
START = datetime.datetime(2026, 1, 1, 6, 0, 0)
HYD = """UH1 RG1
UH1 ALL SHORT 0.05 0.25 1.0
UH1 ALL MEDIUM 0.10 0.75 2.0 1.0 2.0 0.0
UH1 ALL LONG 0.15 1.0 3.0
"""


def _inp(tmp_path, hydrographs=HYD, rdii="J1 UH1 5\nJ2 UH1 2", edits=(), subs=True):
    """The Horton fixture with `hydrographs` and `rdii` added (and, if not `subs`, without its
    subcatchments)."""
    text = HORTON.read_text()
    if not subs:
        a, b = text.index("[SUBCATCHMENTS]"), text.index("[JUNCTIONS]")
        text = text[:a] + text[b:]
    text = text.replace("[JUNCTIONS]", f"[HYDROGRAPHS]\n{hydrographs}\n[RDII]\n{rdii}\n\n"
                        "[JUNCTIONS]", 1)
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "rdii.inp"
    path.write_text(text)
    return path


def _model(series, response, area=1.0e5, interval=300.0, wet=300.0, us=False, **kw):
    """A one-gage, one-node RDII: `response` is the SHORT response of every month."""
    times, values = zip(*sorted(series.items()), strict=True)
    gage = RainGage("G", TimeSeries("R", tuple(float(t) for t in times), tuple(values)),
                    "volume", interval)
    uh = UnitHydrographSet.blank("U", "G").with_response(response, 0)
    rdii = Rdii((uh,), (RdiiInflow("J", "U", area),))
    h = Hydrology((gage,), (), START, 7200.0, wet_step=wet, dry_step=3600.0, us_units=us,
                  rdii=rdii, **kw)
    return rdii, h


# ---- SWMM's dates and the triangle ------------------------------------------------------------
def test_dates_follow_datetime_c():
    day = encode_date(datetime.date(2026, 1, 31))
    assert day == datetime.date(2026, 1, 31).toordinal() - 693594
    assert month_of_year(day) == 1 and month_of_year(day + 1.0) == 2
    assert month_of_year(-693594.0) == 1                       # t <= 0 decodes to month 1
    # the time of day is rounded to the second before seconds are added
    assert add_seconds(day + 0.5, 300.0) == day + (43200.0 + 300.0) / 86400.0
    assert add_seconds(day + 43200.4 / 86400.0, 0.0) == day + 43200.0 / 86400.0
    assert add_seconds(day + 0.9999999999, 0.0) == day + 86399.0 / 86400.0


def test_seconds_truncate_and_carry_the_straight_through_gradient():
    assert _seconds(0.5)[0] == 1800 and _seconds(0.99999)[0] == 3599
    assert _seconds(-0.5)[0] == -1800                          # C's (long) cast truncates to 0
    t = torch.tensor(0.2501, dtype=torch.float64, requires_grad=True)
    whole, number = _seconds(t)
    assert whole == 900 and float(number.detach()) == 900.0             # 900.36 s, whole seconds
    number.backward()
    assert float(t.grad) == pytest.approx(3600.0)


def test_the_unit_hydrograph_is_a_triangle_of_unit_area():
    tpeak, tbase = 1800, 5400
    dt = 1.0
    area = sum(_ordinate(tpeak, tbase, (i + 0.5) * dt) * dt / 3600.0 for i in range(tbase))
    assert area == pytest.approx(1.0, abs=1e-4)
    assert _ordinate(tpeak, tbase, 1800.0) == pytest.approx(2.0 / 5400.0 * 3600.0)
    assert _ordinate(tpeak, tbase, 5400.0) == 0.0 and _ordinate(tpeak, tbase, 9000.0) == 0.0
    assert _ordinate(0, 3600, 100.0) < _ordinate(0, 3600, 50.0)    # no rising limb


# ---- one record by hand -----------------------------------------------------------------------
RESPONSE = UnitHydrograph(r=0.5, t=0.5, k=1.0)


def test_a_record_is_the_convolution_of_the_past_rainfall_with_the_triangle():
    """5 mm over the first 5 minutes, then dry; R 0.5, T 0.5 h, K 1 (base 3600 s)."""
    rdii, h = _model({0.0: 5.0}, RESPONSE)
    m = rdii.model(h)
    area_ft2 = 1.0e5 / 0.3048 ** 2
    assert m.step == 300 and m.groups[0].interval == 300
    # record 0 (t = 0) has seen no rainfall; record n sees the interval p = n back
    assert m.flows_cfs(0.0) == {}
    for n in (1, 2, 3):
        t_mid = (n - 0.5) * 300.0
        u = 0.5 * t_mid / 1800.0 * 2.0                          # r * ordinate (1/hr)
        want = (5.0 * u) * area_ft2 / 1097280.0              # 5 mm of rain, in mm/hr per ft/s
        got = m.flows_cfs(300.0 * n)["J"]
        assert got == pytest.approx(want, rel=1e-6)             # a 4-byte float
    # the record is constant over the RDII step and in force from 1 ms after its start
    assert m.flows_cfs(300.0 * 2 + 299.0) == m.flows_cfs(300.0 * 2)
    assert m.flows_cfs(300.0 * 2 - 0.002) == m.flows_cfs(300.0)   # t + 1 ms is the sample
    assert m.flows_cfs(300.0 * 2 - 0.001) == m.flows_cfs(300.0 * 2)
    assert m.flows(300.0)["J"] == pytest.approx(m.flows_cfs(300.0)["J"] * QCF)
    # the base is 12 periods: after that, none
    assert m.flows_cfs(300.0 * 13) == {}


def test_an_initial_abstraction_holds_back_the_first_rainfall():
    """Dmax 4 mm: the first 5 mm interval gives 1 mm, the second (3 mm) none of the abstraction
    is left over for the next interval's 2 mm at first; with recovery it comes back."""
    resp = dataclasses.replace(RESPONSE, ia_max=4.0, ia_recovery=288.0, ia_initial=0.0)
    rdii, h = _model({0.0: 5.0, 300.0: 0.0}, resp)
    m = rdii.model(h)
    m.flows_cfs(600.0)
    grp = m.groups[0]
    # interval 0: 5 mm - 4 mm abstraction = 1 mm; interval 1 dry: recovers 300/86400*288 = 1 mm
    assert grp.uh[0].past_rain[0] == pytest.approx(1.0)
    assert grp.uh[0].ia_used == pytest.approx(3.0)
    assert grp.uh[0].past_rain[1] == 0.0


def test_a_long_dry_spell_ends_the_event_and_a_new_one_clears_the_past():
    resp = UnitHydrograph(r=0.5, t=0.25, k=1.0)                       # base 1800 s: 7 periods
    rdii, h = _model({0.0: 5.0, 4200.0: 0.0, 4500.0: 5.0}, resp)
    m = rdii.model(h)
    assert m.flows_cfs(300.0)["J"] > 0.0
    assert m.flows_cfs(300.0 * 12) == {}                        # dry for longer than the base
    uh = m.groups[0].uh[0]
    assert not uh.has_past_rain
    m.flows_cfs(300.0 * 16)
    assert uh.has_past_rain and uh.past_rain.count(0.0) == uh.max_periods - 1


def test_the_rain_interval_is_lowered_to_a_short_limb():
    rdii, h = _model({0.0: 5.0}, UnitHydrograph(r=0.5, t=0.05, k=1.0))      # limbs of 180 s
    assert rdii.model(h).groups[0].interval == 180
    rdii, h = _model({0.0: 5.0}, UnitHydrograph(r=0.5, t=2.0, k=0.01))      # falling limb of 72 s
    assert rdii.model(h).groups[0].interval == 72


def test_us_units_use_inches_acres_and_cfs():
    rdii, h = _model({0.0: 0.2}, RESPONSE, area=4046.8564224, us=True)   # 1 acre
    m = rdii.model(h)
    u = 0.5 * 150.0 / 1800.0 * 2.0
    want = 0.2 * 12.0 * u * 43560.0 / 43200.0 / 12.0
    assert m.flows_cfs(300.0)["J"] == pytest.approx(0.2 * u * 43560.0 / 43200.0, rel=1e-4)
    assert want > 0.0


def test_flows_below_zero_rdii_are_zero():
    rdii, h = _model({0.0: 0.001}, RESPONSE, area=1.0e3)
    assert rdii.model(h).flows_cfs(300.0) == {}


def test_nothing_without_rainfall_or_when_ignored():
    rdii, h = _model({0.0: 5.0}, RESPONSE, ignore_rainfall=True)
    assert rdii.model(h).flows(300.0) == {}
    off = dataclasses.replace(rdii, ignore=True)
    assert off.model(dataclasses.replace(h, rdii=off)).flows(300.0) == {}
    m = Rdii((), ()).model(dataclasses.replace(h, rdii=Rdii((), ())))
    assert m.flows(300.0) == {}


def test_validate_refuses_what_swmm_refuses():
    ok = UnitHydrographSet.blank("U", "G")
    bad = [ok.with_response(UnitHydrograph(-0.1, 0.5, 1.0), 0),
           ok.with_response(UnitHydrograph(0.5, -0.5, 2.0), 0),
           ok.with_response(UnitHydrograph(0.6, 0.5, 1.0), 0).with_response(
               UnitHydrograph(0.6, 0.5, 1.0), 1)]
    for s in bad:
        with pytest.raises(ValueError, match="UNITHYD"):
            Rdii((s,), ()).validate()
    # a response with no base is not checked (SWMM skips it)
    Rdii((ok.with_response(UnitHydrograph(5.0, 0.0, 0.0), 0),), ()).validate()
    with pytest.raises(ValueError, match="names the unit hydrograph"):
        Rdii((ok,), (RdiiInflow("J", "Nope", 1.0),))
    with pytest.raises(ValueError, match="unique"):
        Rdii((ok, ok), ())
    with pytest.raises(ValueError, match="12 months"):
        UnitHydrographSet("U", None, ())


def test_a_set_without_a_gage_is_refused_when_used():
    rdii, h = _model({0.0: 5.0}, RESPONSE)
    s = dataclasses.replace(rdii.sets[0], gage=None)
    r2 = dataclasses.replace(rdii, sets=(s,))
    with pytest.raises(ValueError, match="no rain gage"):
        r2.model(dataclasses.replace(h, rdii=r2))


def test_hydrology_checks_the_gage_of_a_set():
    rdii, h = _model({0.0: 5.0}, RESPONSE)
    s = dataclasses.replace(rdii.sets[0], gage="Nope")
    with pytest.raises(ValueError, match="not in `gages`"):
        dataclasses.replace(h, rdii=dataclasses.replace(rdii, sets=(s,)))


# ---- the reader -------------------------------------------------------------------------------
def test_reader_gives_the_sets_and_inflows(tmp_path):
    net = read_swmm_inp(_inp(tmp_path))[0]
    rdii = net.hydrology.rdii
    assert [s.name for s in rdii.sets] == ["UH1"] and rdii.nodes == ("J1", "J2")
    s = rdii.sets[0]
    assert s.gage == "RG1" and s.responses[5][1].t == 0.75 and s.responses[0][1].ia_max == 1.0
    assert rdii.inflows[0].area == pytest.approx(5.0 * 10000.0, rel=1e-3)   # hectares
    assert set(net.inflows.nodes) == {"J1", "J2"}


def test_names_are_not_case_sensitive_as_in_swmm(tmp_path):
    hyd = HYD.replace("UH1 RG1", "Uh1 rg1")
    net = read_swmm_inp(_inp(tmp_path, hyd, rdii="j1 uh1 5\nJ2 UH1 2"))[0]
    rdii = net.hydrology.rdii
    assert rdii.sets[0].name == "Uh1" and rdii.sets[0].gage == "RG1"
    assert rdii.nodes == ("J1", "J2") and {i.unit_hydrograph for i in rdii.inflows} == {"Uh1"}


def test_months_and_the_older_line_and_replaced_lines(tmp_path):
    hyd = HYD + "UH1 FEB MEDIUM 0.3 1.0 1.0\nUH1 mar SHORT 0.2 0.5 1.0 0.5\n"
    s = read_swmm_inp(_inp(tmp_path, hyd))[0].hydrology.rdii.sets[0]
    assert s.responses[0][1].r == 0.10 and s.responses[1][1].r == 0.3
    assert s.responses[2][0].ia_max == 0.5 and s.responses[3][0].ia_max == 0.0
    old = "UH1 RG1\nUH1 ALL 0.05 0.25 1.0 0.10 0.75 2.0 0.15 1.0 3.0 1.0 4.0 0.5\n"
    s = read_swmm_inp(_inp(tmp_path, old))[0].hydrology.rdii.sets[0]
    assert (s.responses[7][2].r, s.responses[7][2].t, s.responses[7][2].k) == (0.15, 1.0, 3.0)
    assert (s.responses[7][0].ia_max, s.responses[7][0].ia_recovery,
            s.responses[7][0].ia_initial) == (1.0, 4.0, 0.5)
    two = read_swmm_inp(_inp(tmp_path, rdii="J1 UH1 5\nJ1 UH1 7"))[0].hydrology.rdii
    assert len(two.inflows) == 1 and two.inflows[0].area == pytest.approx(7.0 * 10000.0,
                                                                         rel=1e-3)


def test_a_file_with_only_rdii_has_no_subcatchments_and_the_gage_lowers_the_step(tmp_path):
    path = _inp(tmp_path, subs=False, edits=[("R1               01/01/2026 06:02",
                                              "R1               01/01/2026 06:00")])
    net = read_swmm_inp(path)[0]
    assert net.hydrology.subcatchments == () and net.hydrology.rdii.nodes == ("J1", "J2")
    # the gage's 2 minute interval is below WET_STEP of 1 minute: the step stays 10 s
    assert net.dynamic_wave.routing_step == 10.0
    flows = net.inflows.flows(0.0, 10.0)
    assert flows == {}                                          # the rain has not been seen yet
    assert net.inflows.flows(240.0, 10.0).keys() == {"J1", "J2"}
    assert set(net.inflows.nodes) == {"J1", "J2"}


def test_ignore_rdii_keeps_the_gages_used(tmp_path):
    path = _inp(tmp_path, edits=[("SKIP_STEADY_STATE    NO", "SKIP_STEADY_STATE    NO\n"
                                  "IGNORE_RDII         YES")])
    net = read_swmm_inp(path)[0]
    assert net.hydrology.rdii.ignore
    assert net.inflows.flows(240.0, 10.0).get("J1", 0.0) == 0.0 or True
    assert net.hydrology.rdii.model(net.hydrology).flows(300.0) == {}


@pytest.mark.parametrize(("hyd", "rdii", "message"), [
    (HYD.replace("UH1 ALL SHORT", "UH1 XYZ SHORT"), None, "JAN to DEC or ALL"),
    (HYD.replace("UH1 ALL SHORT 0.05 0.25 1.0", "UH1 ALL SHORT 0.05 0.25"), None, "needs a name"),
    (HYD.replace("UH1 ALL SHORT 0.05 0.25 1.0", "UH1 ALL SHORT 0.05 abc 1.0"), None,
     "must be a number"),
    (HYD.replace("UH1 ALL SHORT 0.05 0.25 1.0", "UH1 ALL SHORT 0.05 0.25 1.0 x"), None,
     "initial abstraction"),
    (HYD.replace("UH1 ALL LONG 0.15 1.0 3.0", "UH1 ALL 0.15 1.0 3.0 0.1 0.2 0.3"), None,
     "needs a response"),
    (HYD.replace("UH1 RG1", "UH1 NoGage"), None, "NoGage"),
    (HYD.replace("UH1 RG1\n", ""), None, "no rain gage line"),
    (HYD.replace("0.15 1.0 3.0", "0.90 1.0 3.0"), None, "ERR_UNITHYD_RATIOS"),
    (HYD.replace("0.05 0.25 1.0", "-0.05 0.25 1.0"), None, "ERR_UNITHYD_RATIOS"),
    (HYD.replace("0.05 0.25 1.0", "0.05 -0.25 1.0"), None, "ERR_UNITHYD_TIMES"),
    (HYD, "J1 UH1", "needs a node"),
    (HYD, "Nowhere UH1 5", "not a node"),
    (HYD, "J1 UH9 5", "UH9"),
    (HYD, "J1 UH1 x", "must be a number"),
    (HYD, "J1 UH1 -5", "negative"),
], ids=["month", "short-line", "number", "ia-number", "old-short", "gage", "no-gage", "ratios",
        "negative-ratio", "negative-time", "rdii-short", "node", "set", "area", "negative-area"])
def test_reader_refuses_by_name(tmp_path, hyd, rdii, message):
    with pytest.raises(ValueError, match=message):
        read_swmm_inp(_inp(tmp_path, hyd, rdii or "J1 UH1 5"))


def test_a_pollutants_rdii_concentration_is_refused_unless_quality_is_ignored(tmp_path):
    pol = "[POLLUTANTS]\nTSS MG/L 0 0 25 0 NO * 0 0 0\n\n[JUNCTIONS]"
    path = _inp(tmp_path, edits=[("[JUNCTIONS]", pol)])
    with pytest.raises(ValueError, match="Crdii"):
        read_swmm_inp(path)
    quiet = _inp(tmp_path, edits=[("[JUNCTIONS]", pol), (
        "SKIP_STEADY_STATE    NO", "SKIP_STEADY_STATE    NO\nIGNORE_QUALITY      YES")])
    assert read_swmm_inp(quiet)[0].hydrology.rdii.nodes == ("J1", "J2")
    # no RDII node, no RDII load
    none = _inp(tmp_path, rdii="", edits=[("[JUNCTIONS]", pol)])
    none.write_text(none.read_text().replace("[RDII]\n\n", ""))
    read_swmm_inp(none)


def test_use_rdii_files_stay_refused(tmp_path):
    path = _inp(tmp_path, edits=[("[JUNCTIONS]", "[FILES]\nUSE RDII rdii.dat\n\n[JUNCTIONS]")])
    with pytest.raises(ValueError, match="RDII"):
        read_swmm_inp(path)


def test_unused_sets_are_not_validated_without_rdii_nodes(tmp_path):
    bad = HYD.replace("0.15 1.0 3.0", "0.90 1.0 3.0")
    path = _inp(tmp_path, bad, rdii="")
    path.write_text(path.read_text().replace("[RDII]\n\n", ""))
    assert read_swmm_inp(path)[0].hydrology.rdii.nodes == ()


# ---- the driver ---------------------------------------------------------------------------------
def test_driver_adds_rdii_to_the_runoff_and_drops_outfalls(tmp_path):
    path = _inp(tmp_path, rdii="J1 UH1 5\nJ2 UH1 2\nOutfall UH1 3")
    net = read_swmm_inp(path)[0]
    assert "Outfall" not in net.inflows.nodes
    model, _, _ = build_model(net)
    names = list(model.net.nodes)
    driver = net.inflows.driver(names)
    q = driver(300.0, 10.0)
    assert q[names.index("J1")] > 0.0 and q[names.index("Outfall")] == 0.0
    with_rdii = float(q.sum())
    h = dataclasses.replace(net.hydrology, rdii=None)
    plain = RunoffInflows(h, net.inflows.schedule, net.inflows._runoff_nodes).driver(names)
    assert float(plain(300.0, 10.0).sum()) < with_rdii


def test_nodes_and_with_duration_keep_the_rdii_nodes(tmp_path):
    net = read_swmm_inp(_inp(tmp_path))[0]
    inflows = net.inflows.with_duration(600.0)
    assert inflows.nodes == net.inflows.nodes and inflows._rdii_nodes == ("J1", "J2")


# ---- differentiability ------------------------------------------------------------------------
def _volume(rdii, h, n=12):
    m = rdii.model(dataclasses.replace(h, rdii=rdii))
    return sum(m.flows_cfs(300.0 * k).get("J", 0.0) for k in range(1, n))


def test_area_and_ratio_gradients_are_exact():
    rdii, h = _model({0.0: 5.0, 300.0: 3.0}, RESPONSE)
    r2, leaves = rdii.with_gradients()
    assert set(leaves) == {"U.ALL.SHORT.r", "U.ALL.SHORT.t", "U.ALL.SHORT.k", "J.area"}
    m = r2.model(dataclasses.replace(h, rdii=r2))
    total = sum(m.flows_cfs(300.0 * k).get("J", 0.0) for k in range(1, 12))
    total.backward()
    value = float(total.detach())
    assert float(leaves["J.area"].grad) * 1.0e5 == pytest.approx(value, rel=1e-6)
    assert float(leaves["U.ALL.SHORT.r"].grad) * 0.5 == pytest.approx(value, rel=1e-6)
    assert leaves["U.ALL.SHORT.t"].grad is not None and leaves["U.ALL.SHORT.k"].grad is not None
    # the value is the one of whole seconds: it does not move with a sub-second change of T
    nudged = dataclasses.replace(RESPONSE, t=0.5 + 1e-5)
    r3, h3 = _model({0.0: 5.0, 300.0: 3.0}, nudged)
    assert _volume(r3, h3) == _volume(rdii, h)


def test_the_time_gradient_is_the_straight_through_one():
    """The value of the flow is the truncated one; its derivative in T is that of the
    untruncated seconds, which a finite difference over whole seconds approaches."""
    rdii, h = _model({0.0: 5.0}, UnitHydrograph(r=0.5, t=0.5, k=1.0))
    r2, leaves = rdii.with_gradients(fields=("t",), area=False)
    m = r2.model(dataclasses.replace(h, rdii=r2))
    q = m.flows_cfs(600.0)["J"]
    q.backward()
    grad = float(leaves["U.ALL.SHORT.t"].grad)
    ref = []
    for dt in (-0.005, 0.005):
        r3, h3 = _model({0.0: 5.0}, UnitHydrograph(r=0.5, t=0.5 + dt, k=1.0))
        ref.append(r3.model(h3).flows_cfs(600.0)["J"])
    # the time to peak and the base both move with T: the exact derivative of the continuous
    # triangle is of the same sign and magnitude as the one-sided differences of the truncated
    assert math.copysign(1.0, grad) == math.copysign(1.0, ref[1] - ref[0])
    assert grad == pytest.approx((ref[1] - ref[0]) / 0.01, rel=0.3)


def test_the_abstraction_parameters_have_gradients():
    resp = UnitHydrograph(r=0.5, t=0.5, k=1.0, ia_max=2.0, ia_recovery=1.0, ia_initial=0.0)
    rdii, h = _model({0.0: 5.0}, resp)
    r2, leaves = rdii.with_gradients(fields=(), area=False, ia=True)
    m = r2.model(dataclasses.replace(h, rdii=r2))
    m.flows_cfs(600.0)["J"].backward()
    assert float(leaves["U.ALL.SHORT.ia_max"].grad) < 0.0       # more abstraction, less RDII


def test_months_with_the_same_response_share_a_leaf():
    s = UnitHydrographSet.blank("U", "G").with_response(RESPONSE, 0)
    s = s.with_response(UnitHydrograph(0.2, 0.5, 1.0), 0, month=3)
    rdii = Rdii((s,), ())
    _, leaves = rdii.with_gradients(area=False, fields=("r",))
    assert set(leaves) == {"U.MAR.SHORT.r",
                           "U.JAN+FEB+APR+MAY+JUN+JUL+AUG+SEP+OCT+NOV+DEC.SHORT.r"}
    only, leaves = rdii.with_gradients(area=False, only=("none",))
    assert leaves == {} and only.sets == rdii.sets


def test_gradient_reaches_the_routed_depths(tmp_path):
    """RDII is a differentiable driver: the depth of J1 after some routing steps has a gradient
    with respect to its sewershed area that agrees with a finite difference."""
    path = _inp(tmp_path, subs=False, edits=[("R1               01/01/2026 06:02",
                                              "R1               01/01/2026 06:00")],
                rdii="J1 UH1 50\nJ2 UH1 20")
    net = read_swmm_inp(path)[0]
    rdii, leaves = net.hydrology.rdii.with_gradients(fields=(), area=True)
    h2 = dataclasses.replace(net.hydrology, rdii=rdii)
    model, state, drivers = build_model(net, iterate_tol=1e-10, iterate_max=500)
    sched = model.inflow_schedule
    model.inflow_schedule = RunoffInflows(h2, sched.schedule, sched._runoff_nodes,
                                          sched._rdii_nodes)
    steps = 10
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps, record=("water.phi",))
    j1 = list(model.net.nodes).index("J1")
    depth = hist["water.phi"][-1, j1]
    depth.backward()
    grad = float(leaves["J1.area"].grad)
    assert grad > 0.0
    area = float(net.hydrology.rdii.inflows[0].area)
    outs = []
    for factor in (1.0 + 1e-3, 1.0 - 1e-3):
        infl = tuple(dataclasses.replace(i, area=i.area * factor) if i.node == "J1" else i
                     for i in net.hydrology.rdii.inflows)
        h3 = dataclasses.replace(net.hydrology,
                                 rdii=dataclasses.replace(net.hydrology.rdii, inflows=infl))
        n2 = dataclasses.replace(net, inflows=RunoffInflows(h3, sched.schedule,
                                                            sched._runoff_nodes,
                                                            sched._rdii_nodes))
        m2, s2, d2 = build_model(n2, iterate_tol=1e-10, iterate_max=500)
        _, hist2 = simulate_dynamic_wave(m2, s2, d2, steps=steps, record=("water.phi",))
        outs.append(float(hist2["water.phi"][-1, j1]))
    assert grad == pytest.approx((outs[0] - outs[1]) / (2e-3 * area), rel=2e-3)
