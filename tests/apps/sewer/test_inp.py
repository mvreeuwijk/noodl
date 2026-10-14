"""The documented SWMM .inp subset."""

import dataclasses
import math
from pathlib import Path

import pytest

from noodl.apps.sewer.inp import read_swmm_inp
from noodl.apps.sewer.network import Exfiltration

DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"


def test_the_committed_fixture_round_trips_to_the_reference_tree():
    net, inflows, pollutants = read_swmm_inp(DATA / "tree_steady.inp")
    net.validate()
    assert [m.name for m in net.manholes] == ["J1", "J2", "J5", "J3", "J4"]
    assert [o.name for o in net.outfalls] == ["Outfall"]
    assert [p.name for p in net.pipes] == ["C1", "C2", "C3", "C4", "C5"]
    # 5 decimals, not 10: the reader's own
    # docstring and `test_conduits_are_normalised_upstream_to_downstream` below both pin the
    # slope to the 3-D CHORD formula (`dy / sqrt(L**2 - dy**2)`), which on this fixture's
    # exact elevations and length gives 0.0100005.../0.0050000625..., not 0.01/0.005 to 6 or
    # 10 decimals -- a real number, not a rounding artefact (confirmed with `decimal.Decimal`).
    assert [round(p.slope, 5) for p in net.pipes] == [0.01, 0.01, 0.005, 0.01, 0.005]
    assert [p.diameter for p in net.pipes] == [0.30, 0.30, 0.45, 0.30, 0.45]
    assert {m.name: m.inflow for m in net.manholes} == {
        "J1": 0.05, "J2": 0.08, "J5": 0.03, "J3": 0.0, "J4": 0.0
    }
    assert pollutants == {}
    assert inflows == {}


def test_the_pollutant_variant_carries_the_tracer():
    net, inflows, pollutants = read_swmm_inp(DATA / "tree_kinwave_pollut.inp")
    assert set(pollutants) == {"Tracer"}
    # MG/L is converted to kg/m3 by 1e-3, and Kdecay 1/day to 1/s
    assert pollutants["Tracer"]["decay"] == pytest.approx(5.0 / 86400.0, rel=1e-14)
    assert inflows == {"J1": {"Tracer": pytest.approx(0.1, rel=1e-14)}}


def test_a_nonzero_initial_pollutant_concentration_is_refused_by_name(tmp_path):
    """Cinit sets the concentration of the water a run starts with; the model starts clean,
    so a nonzero Cinit is refused naming the pollutant and the line. The fields only runoff,
    groundwater, RDII or dry-weather flow would read (refused sections) are not checked."""
    text = (DATA / "tree_kinwave_pollut.inp").read_text()
    old = ("Tracer           MG/L   0.0        0.0        0.0        5.0        NO         *"
           "                0.0        0.0        0.0")
    assert text.count(old) == 1
    path = tmp_path / "cinit.inp"
    path.write_text(text.replace(old, old[:-3] + "2.5"))
    with pytest.raises(ValueError, match=r"line 51: pollutant 'Tracer' has a nonzero Cinit"):
        read_swmm_inp(path)
    cdwf = old[:-14] + "7.0        0.0"                  # Cdwf 7: only [DWF] reads it
    path.write_text(text.replace(old, cdwf))
    read_swmm_inp(path)


def test_conduit_offsets_shift_the_effective_invert_under_link_offsets_depth(tmp_path):
    """Under `LINK_OFFSETS DEPTH` (the fixture's own default), a conduit's
    InOffset/OutOffset add to the NODE invert at each end before the slope is computed --
    untested by the committed fixtures, which all carry zero offsets. Gives C1 an
    OutOffset of 0.5 m at J3 (invert 10.0), so its effective downstream elevation is 10.5,
    not 10.0."""
    original = (DATA / "tree_steady.inp").read_text()
    old_line = next(line for line in original.splitlines() if line.startswith("C1 "))
    fields = old_line.split()
    fields[6] = "0.5"  # OutOffset (name, node1, node2, length, n, InOffset, OutOffset, ...)
    text = original.replace(old_line, " ".join(fields))
    path = tmp_path / "offset.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    pipe = next(p for p in net.pipes if p.name == "C1")
    fall = 12.0 - (10.0 + 0.5)
    assert pipe.slope == pytest.approx(fall / (200.0**2 - fall**2) ** 0.5, rel=1e-12)


def test_conduits_are_normalised_upstream_to_downstream():
    """C1 is written J1 -> J3 with inverts 12.0 and 10.0, so it already runs downhill; the
    reader still recomputes the slope from the inverts and the 3-D chord length."""
    net, _, _ = read_swmm_inp(DATA / "tree_steady.inp")
    pipe = next(p for p in net.pipes if p.name == "C1")
    assert (pipe.u, pipe.v) == ("J1", "J3")
    assert pipe.slope == pytest.approx(2.0 / (200.0**2 - 2.0**2) ** 0.5, rel=1e-12)


def test_an_unknown_section_with_content_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text() + "\n[PUMPS]\nP1 J1 J3 HEAD C1\n"
    path = tmp_path / "pumps.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"\[PUMPS\]"):
        read_swmm_inp(path)


def test_an_inflow_naming_an_undefined_time_series_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        'J1               FLOW             ""               FLOW     1.0      1.0      0.05',
        "J1               FLOW             TS1              FLOW     1.0      1.0      0.05",
    )
    path = tmp_path / "ts.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"time series 'TS1', which has no \[TIMESERIES\]"):
        read_swmm_inp(path)


def test_a_non_circular_cross_section_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "C1               CIRCULAR", "C1               RECT_CLOSED"
    )
    path = tmp_path / "rect.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="shape other than CIRCULAR"):
        read_swmm_inp(path)


def test_a_conduit_with_zero_fall_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "J3               10.0       5", "J3               12.0       5"
    )
    path = tmp_path / "flat.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="zero or adverse fall"):
        read_swmm_inp(path)


def test_a_dwf_flow_is_read_and_a_pollutant_dwf_line_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text() + "\n[DWF]\nJ3 FLOW 0.01\n"
    path = tmp_path / "dwf.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    assert next(m for m in net.manholes if m.name == "J3").inflow == 0.01
    path.write_text(text + "J3 BOD 5\n")
    with pytest.raises(ValueError, match="only the dry-weather FLOW"):
        read_swmm_inp(path)


def test_the_kinwave_variant_records_its_routing():
    net, _, _ = read_swmm_inp(DATA / "tree_kinwave.inp")
    assert net.routing == "kinematic_wave"


def _dyn(tmp_path, edits=()):
    text = (DATA / "tree_kinwave.inp").read_text()
    text = text.replace("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE")
    text = text.replace("VARIABLE_STEP        0.75", "VARIABLE_STEP        0")
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / "dyn.inp"
    path.write_text(text)
    return path


def test_dynwave_options_are_read(tmp_path):
    net, _, _ = read_swmm_inp(_dyn(tmp_path))
    assert net.routing == "dynamic_wave"
    o = net.dynamic_wave
    assert o.routing_step == 5.0 and o.inertial_damping == "partial"
    assert o.normal_flow_limited == "both" and o.min_surface_area == 12.566
    assert o.head_tolerance == 0.0015 and o.max_trials == 8


def test_dynwave_routing_step_accepts_a_clock_time(tmp_path):
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [("ROUTING_STEP         5",
                                               "ROUTING_STEP         00:00:30")]))
    assert net.dynamic_wave.routing_step == 30.0


def test_dynwave_reads_the_variable_step_options(tmp_path):
    path = _dyn(tmp_path, [("VARIABLE_STEP        0", "VARIABLE_STEP        0.75"),
                           ("MINIMUM_STEP         0.5", "MINIMUM_STEP         0.25")])
    net, _, _ = read_swmm_inp(path)
    assert net.dynamic_wave.variable_step == 0.75
    assert net.dynamic_wave.minimum_step == 0.25
    # absent, the option is SWMM's default 0.75 (project.c:854), whatever the dataclass says
    lines = _dyn(tmp_path).read_text().splitlines(keepends=True)
    absent = "".join(x for x in lines if not x.startswith("VARIABLE_STEP"))
    (tmp_path / "absent.inp").write_text(absent)
    net, _, _ = read_swmm_inp(tmp_path / "absent.inp")
    assert net.dynamic_wave.variable_step == 0.75


def test_dynwave_routing_step_is_lowered_to_the_wet_step(tmp_path):
    """project.c:246-250: ROUTING_STEP above WET_STEP becomes WET_STEP (default 300 s)."""
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [("ROUTING_STEP         5",
                                               "ROUTING_STEP         90")]))
    assert net.dynamic_wave.routing_step == 60.0           # the fixture's WET_STEP 00:01:00
    lines = _dyn(tmp_path, [("ROUTING_STEP         5", "ROUTING_STEP         400")]
                 ).read_text().splitlines(keepends=True)
    (tmp_path / "nowet.inp").write_text("".join(x for x in lines
                                                if not x.startswith("WET_STEP")))
    net, _, _ = read_swmm_inp(tmp_path / "nowet.inp")
    assert net.dynamic_wave.routing_step == 300.0           # SWMM's default WET_STEP
    net, _, _ = read_swmm_inp(_dyn(tmp_path))
    assert net.dynamic_wave.routing_step == 5.0


@pytest.mark.parametrize("old,new,key", [
    ("VARIABLE_STEP        0", "VARIABLE_STEP        2.5", "VARIABLE_STEP"),
    ("MINIMUM_STEP         0.5", "MINIMUM_STEP         -1", "MINIMUM_STEP"),
])
def test_dynwave_refuses_a_bad_variable_step_option_with_its_line(tmp_path, old, new, key):
    with pytest.raises(ValueError, match=rf"line \d+: {key}"):
        read_swmm_inp(_dyn(tmp_path, [(old, new)]))


@pytest.mark.parametrize("old,new,match", [
    ("LENGTHENING_STEP     0", "LENGTHENING_STEP     -5", r"line \d+: LENGTHENING_STEP"),
    ("J1               12.0       5          0", "J1               12.0       5          -0.2",
     "negative initial depth"),
    ("C1               CIRCULAR     0.30             0          0          0          1",
     "C1               CIRCULAR     0.30             0          0          0          2.5",
     "whole number"),
    ("C1               CIRCULAR     0.30             0          0          0          1",
     "C1               CIRCULAR     0.30             0          0          0          128",
     "from 1 to 127"),
    ("Outfall          8.0        FREE                        NO",
     "Outfall          8.0        TIDAL      Tide1            NO", "Tide1"),
    ("Outfall          8.0        FREE                        NO",
     "Outfall          8.0        TIDAL", "needs a tide curve"),
    ("Outfall          8.0        FREE                        NO",
     "Outfall          8.0        TIMESERIES", "needs a time series"),
])
def test_dynwave_refusals(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read_swmm_inp(_dyn(tmp_path, [(old, new)]))


TIDE = ("[CURVES]\n;;Name Type x y\nTide1 TIDAL 0 8.1 6 8.4\nTide1 12 8.2 24 8.1\n\n"
        "[OUTFALLS]\n")


def test_a_tidal_outfall_is_read_with_its_curve_and_gate(tmp_path):
    """TIDAL (node.c:1357-1380): field 3 names a [CURVES] TIDAL curve of (hour, stage
    elevation) pairs over any number of lines; the gate flag follows it."""
    path = _dyn(tmp_path, [
        ("Outfall          8.0        FREE                        NO",
         "Outfall          8.0        TIDAL      Tide1            YES"),
        ("[OUTFALLS]\n", TIDE)])
    net, _, _ = read_swmm_inp(path)
    (outfall,) = net.outfalls
    assert outfall.kind == "tidal" and outfall.flap_gate
    assert outfall.tide == ((0.0, 8.1), (6.0, 8.4), (12.0, 8.2), (24.0, 8.1))


def test_a_timeseries_outfall_takes_its_stage_from_its_series(tmp_path):
    """TIMESERIES (node.c:1385-1389, 1454-1458): the reader reads the named series, in
    metres, as the outfall's default stage; it is held at its ends (table_tseriesLookup with
    extend TRUE, table.c:779) where an inflow series is 0 outside. A series no inflow or
    outfall names is not read."""
    edits = [("Outfall          8.0        FREE                        NO",
              "Outfall          8.0        TIMESERIES Tide1            NO"),
             ("[REPORT]", "[TIMESERIES]\nTide1 0:00 8.1\nTide1 6:00 8.4\n\n[REPORT]")]
    net, _, _ = read_swmm_inp(_dyn(tmp_path, edits))
    (outfall,) = net.outfalls
    assert (outfall.kind, outfall.series, outfall.stage) == ("timeseries", "Tide1", None)
    series = outfall.stage_series
    assert series.times == (0.0, 21600.0) and series.values == (8.1, 8.4)
    assert [series.held(t) for t in (-5.0, 0.0, 10800.0, 21600.0, 1e6)] == pytest.approx(
        [8.1, 8.1, 8.25, 8.4, 8.4], abs=1e-12)
    assert series(1e6) == 0.0                       # an inflow series is 0 beyond its end
    edits[1] = ("[REPORT]", "[TIMESERIES]\nOther 0:00 8.1\n\n[REPORT]")
    with pytest.raises(ValueError, match=r"Tide1.*no \[TIMESERIES\] entry"):
        read_swmm_inp(_dyn(tmp_path, edits))


def test_a_timeseries_outfall_stage_is_converted_to_metres(tmp_path):
    """A US-unit file gives the series in feet; `stage_series` is in metres."""
    text = _dyn(tmp_path, [
        ("Outfall          8.0        FREE                        NO",
         "Outfall          8.0        TIMESERIES Tide1            NO"),
        ("[REPORT]", "[TIMESERIES]\nTide1 0:00 8.0\nTide1 1:00 10.0\n\n[REPORT]"),
        ("FLOW_UNITS           CMS", "FLOW_UNITS           CFS")]).read_text()
    (tmp_path / "us.inp").write_text(text)
    (outfall,) = read_swmm_inp(tmp_path / "us.inp")[0].outfalls
    assert outfall.stage_series.values == pytest.approx((8.0 * 0.3048, 10.0 * 0.3048))


def test_the_stage_series_is_the_default_outfall_stage_and_an_argument_overrides_it(tmp_path):
    """`build_model` and `simulate_dynamic_wave` need no `outfall_stage` for a file's
    TIMESERIES outfall; naming it replaces the series."""
    from noodl.apps.sewer import build_model
    path = _dyn(tmp_path, [
        ("Outfall          8.0        FREE                        NO",
         "Outfall          8.0        TIMESERIES Tide1            NO"),
        ("[REPORT]", "[TIMESERIES]\nTide1 0:00 8.1\nTide1 1:00 8.5\n\n[REPORT]")])
    net, _, _ = read_swmm_inp(path)
    model, _, _ = build_model(net)
    hyd = model.dynamic_wave
    assert hyd.outfall_stage(1800.0, model.outfall_stage).tolist() == pytest.approx([8.3])
    model, _, _ = build_model(net, outfall_stage={"Outfall": 9.0})
    assert hyd.outfall_stage(1800.0, model.outfall_stage).tolist() == [9.0]


def test_a_tide_curve_must_be_a_tidal_curve(tmp_path):
    path = _dyn(tmp_path, [
        ("Outfall          8.0        FREE                        NO",
         "Outfall          8.0        TIDAL      Tide1            NO"),
        ("[OUTFALLS]\n", TIDE.replace("TIDAL", "RATING"))])
    with pytest.raises(ValueError, match="Tide1.*RATING"):
        read_swmm_inp(path)


C1_LINE = ("C1               J1               J3               200        0.013      0"
           "          0          0")


def _c1(init_flow, max_flow):
    """The tree's C1 line with an InitFlow and a MaxFlow (m3/s)."""
    return ("C1               J1               J3               200        0.013      0"
            f"          0          {init_flow}       {max_flow}")


def test_dynwave_reads_initial_conditions_limits_barrels_and_gates(tmp_path):
    """InitDepth (junction), InitFlow and MaxFlow (conduit), barrels, [LOSSES] flap gate and
    the outfall's Gated flag are physics under DYNWAVE and are read, not refused."""
    path = _dyn(tmp_path, [
        ("J1               12.0       5          0", "J1               12.0       5          0.2"),
        (C1_LINE, _c1(0.01, 0.05)),
        ("C2               CIRCULAR     0.30             0          0          0          1",
         "C2               CIRCULAR     0.30             0          0          0          3"),
        ("Outfall          8.0        FREE                        NO",
         "Outfall          8.0        FREE                        YES"),
        ("[INFLOWS]", "[LOSSES]\nC3  0  0  0  YES  0\nC4  0  0  0  NO\n\n[INFLOWS]"),
    ])
    net, _, _ = read_swmm_inp(path)
    assert next(m for m in net.manholes if m.name == "J1").initial_depth == 0.2
    c1, c2, c3, c4 = (next(p for p in net.pipes if p.name == n) for n in
                      ("C1", "C2", "C3", "C4"))
    assert (c1.initial_flow, c1.max_flow, c1.barrels) == (0.01, 0.05, 1)
    assert c2.barrels == 3 and c2.max_flow is None and c2.initial_flow == 0.0
    assert c3.flap_gate and not c3.flap_gate_reversed and not c4.flap_gate
    assert net.outfalls[0].flap_gate


def test_a_reversed_conduit_flips_its_initial_flow_and_gate_direction(tmp_path):
    """SWMM's conduit_reverse negates q0 and the link direction (link.c:1189-1190): C4 is
    adverse (J5 8 m under J4 9 m) and written J5 -> J4, so it is J4 -> J5 for noodl; an
    InitFlow of 0.01 from J5 to J4 is -0.01 along J4 -> J5, and its gate, which passes the
    file's J5 -> J4 direction, passes v -> u."""
    path = _dyn(tmp_path, [
        ("J5               11.0", "J5               8.0"),
        ("C4               J5               J4               200        0.013      0"
         "          0", "C4               J5               J4               200        0.013"
         "      0          0          0.01"),
        ("[INFLOWS]", "[LOSSES]\nC4  0  0  0  YES\n\n[INFLOWS]"),
    ])
    net, _, _ = read_swmm_inp(path)
    c4 = next(p for p in net.pipes if p.name == "C4")
    assert (c4.u, c4.v) == ("J4", "J5")
    assert c4.initial_flow == -0.01
    assert c4.flap_gate and c4.flap_gate_reversed


def test_dynwave_reads_lengthening_with_the_files_own_routing_step(tmp_path):
    """LENGTHENING_STEP is read, and conduits are lengthened with the file's ROUTING_STEP
    even when WET_STEP lowers the routing step afterwards (SWMM validates the conduits first,
    project.c:238-250)."""
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [
        ("LENGTHENING_STEP     0", "LENGTHENING_STEP     10"),
        ("ROUTING_STEP         5", "ROUTING_STEP         90")]))
    opts = net.dynamic_wave
    assert (opts.lengthening_step, opts.routing_step, opts.lengthening_routing_step) == (
        10.0, 60.0, 90.0)


@pytest.mark.parametrize("what,edit,match", [
    ("a negative entry loss", "C1  -0.5  0  0", "negative entry loss"),
    ("a negative exit loss", "C1  0  -0.5  0", "negative exit loss"),
    ("a negative average loss", "C1  0  0  -0.5", "negative average loss"),
    ("a bad flap word", "C1  0  0  0  MAYBE", "YES or NO"),
    ("an unknown link", "Cx  0  0  0  YES", "Cx.*not a conduit"),
])
def test_losses_beyond_the_flap_gate_are_refused(tmp_path, what, edit, match):
    path = _dyn(tmp_path, [("[INFLOWS]", f"[LOSSES]\n{edit}\n\n[INFLOWS]")])
    with pytest.raises(ValueError, match=match):
        read_swmm_inp(path)


@pytest.mark.parametrize("edits,match", [
    ([("J1               12.0       5          0", "J1               12.0       5          0.2")],
     "initial_depth"),
    ([(C1_LINE, _c1(0.01, 0))],
     "initial_flow"),
    ([(C1_LINE, _c1(0, 0.05))],
     None),
    ([("[INFLOWS]", "[LOSSES]\nC1  0  0  0  YES\n\n[INFLOWS]")], "flap_gate"),
    ([("Outfall          8.0        FREE                        NO",
       "Outfall          8.0        FREE                        YES")], "flap gate"),
])
def test_kinwave_refuses_dynamic_wave_physics_by_name(tmp_path, edits, match):
    """STEADY and KINWAVE do not model InitDepth, InitFlow, flap gates; a set value is refused
    by name rather than skipped (MaxFlow is a kinematic-wave limit too and is read, but
    only dynamic wave applies it here: it is accepted without effect only when 0 = none)."""
    text = (DATA / "tree_kinwave.inp").read_text()
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / "k.inp"
    path.write_text(text)
    if match is None:
        with pytest.raises(ValueError, match="max_flow"):
            read_swmm_inp(path)
    else:
        with pytest.raises(ValueError, match=match):
            read_swmm_inp(path)


def test_dynwave_keeps_offsets_and_floors_a_flat_fall(tmp_path):
    path = _dyn(tmp_path, [
        ("J5               11.0", "J5               9.0"),          # C4 J5 -> J4 flat
        ("C1               J1               J3               200        0.013      0          0",
         "C1               J1               J3               200        0.013      0.1        0.2"),
    ])
    net, _, _ = read_swmm_inp(path)
    c1 = next(p for p in net.pipes if p.name == "C1")
    assert (c1.inlet_offset, c1.outlet_offset) == (0.1, 0.2)
    c4 = next(p for p in net.pipes if p.name == "C4")
    assert (c4.u, c4.v) == ("J5", "J4")                 # flat: file orientation kept
    dz = 0.001 * 0.3048                                 # link.c MIN_DELTA_Z, ft -> m
    assert c4.slope == pytest.approx(dz / math.sqrt(200.0**2 - dz**2), rel=1e-12)


def test_dynwave_reverses_only_an_adverse_conduit_and_swaps_offsets(tmp_path):
    path = _dyn(tmp_path, [
        ("J5               11.0", "J5               8.0"),          # C4 J5 -> J4 adverse
        ("C4               J5               J4               200        0.013      0          0",
         "C4               J5               J4               200        0.013      0.1        0.2"),
    ])
    net, _, _ = read_swmm_inp(path)
    c4 = next(p for p in net.pipes if p.name == "C4")
    assert (c4.u, c4.v) == ("J4", "J5")
    assert (c4.inlet_offset, c4.outlet_offset) == (0.2, 0.1)
    assert c4.slope > 0


def test_dynwave_min_slope_floors_the_slope(tmp_path):
    path = _dyn(tmp_path, [("MIN_SLOPE            0", "MIN_SLOPE            5")])
    net, _, _ = read_swmm_inp(path)
    assert all(p.slope >= 0.05 for p in net.pipes)


def test_fixed_outfall_stage_is_read(tmp_path):
    path = _dyn(tmp_path, [("Outfall          8.0        FREE                        NO",
                            "Outfall          8.0        FIXED      8.2              NO")])
    net, _, _ = read_swmm_inp(path)
    assert net.outfalls[0].kind == "fixed" and net.outfalls[0].stage == 8.2


def test_barrels_above_one_are_refused_for_kinwave_too(tmp_path):
    text = (DATA / "tree_kinwave.inp").read_text().replace(
        "C1               CIRCULAR     0.30             0          0          0          1",
        "C1               CIRCULAR     0.30             0          0          0          2")
    path = tmp_path / "k.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="barrels"):
        read_swmm_inp(path)


def _elevation_variant(tmp_path, dynwave):
    """The tree fixture with C1 offsets 0.1/0.2 m, as DEPTH and as ELEVATION files."""
    old = ("C1               J1               J3               200        0.013      0"
           "          0")
    depth = (DATA / "tree_kinwave.inp").read_text().replace(
        old, old.replace("0          0", "0.1        0.2"))
    elev = depth.replace("LINK_OFFSETS         DEPTH", "LINK_OFFSETS         ELEVATION")
    for pipe, (z1, z2) in {"C1": (12.1, 10.2), "C2": (12.0, 10.0), "C3": (10.0, 9.0),
                           "C4": (11.0, 9.0), "C5": (9.0, 8.0)}.items():
        lines = elev.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith(pipe + " ") and "0.013" in ln)
        head = lines[i].split("0.013")[0] + "0.013"
        lines[i] = f"{head}      {z1}       {z2}       0          0"
        elev = chr(10).join(lines)
    if dynwave:
        depth, elev = (t.replace("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE")
                       .replace("VARIABLE_STEP        0.75", "VARIABLE_STEP        0")
                       for t in (depth, elev))
    out = []
    for name, text in (("d.inp", depth), ("e.inp", elev)):
        (tmp_path / name).write_text(text)
        out.append(read_swmm_inp(tmp_path / name)[0])
    return out


@pytest.mark.parametrize("dynwave", [False, True])
def test_elevation_offsets_are_stored_above_the_node_invert(tmp_path, dynwave):
    depth, elev = _elevation_variant(tmp_path, dynwave)
    c1d = next(p for p in depth.pipes if p.name == "C1")
    c1e = next(p for p in elev.pipes if p.name == "C1")
    assert (c1e.inlet_offset, c1e.outlet_offset) == pytest.approx((0.1, 0.2), abs=1e-12)
    assert (c1e.inlet_offset, c1e.outlet_offset) == pytest.approx(
        (c1d.inlet_offset, c1d.outlet_offset), abs=1e-12)
    assert c1e.slope == pytest.approx(c1d.slope, rel=1e-12)


def test_dynwave_only_options_do_not_raise_for_kinwave(tmp_path):
    text = (DATA / "tree_kinwave.inp").read_text().replace(
        "ROUTING_STEP         5", "ROUTING_STEP         bad:time:x")
    path = tmp_path / "k.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    assert net.routing == "kinematic_wave"


def test_dynwave_surcharge_method_slot_is_read(tmp_path):
    path = _dyn(tmp_path, [("SKIP_STEADY_STATE    NO",
                            "SKIP_STEADY_STATE    NO\nSURCHARGE_METHOD    SLOT")])
    net, _, _ = read_swmm_inp(path)
    assert net.dynamic_wave.surcharge_method == "slot"
    assert read_swmm_inp(_dyn(tmp_path))[0].dynamic_wave.surcharge_method == "extran"


def test_junction_surcharge_depth_is_read(tmp_path):
    """SWMM's junction field 4, SurDepth (m): the depth a junction may pressurise above its
    MaxDepth before it floods (node.c:149-154). Absent, it is 0; a negative one is refused
    by name, as SWMM refuses it (node.c:638-642)."""
    path = _dyn(tmp_path, [("J2               12.0       5          0          0",
                            "J2               12.0       5          0          1.5")])
    net, _, _ = read_swmm_inp(path)
    sur = {m.name: m.surcharge_depth for m in net.manholes}
    assert sur == {"J1": 0.0, "J2": 1.5, "J5": 0.0, "J3": 0.0, "J4": 0.0}
    path = _dyn(tmp_path, [("J2               12.0       5          0          0",
                            "J2               12.0       5          0          -0.1")])
    with pytest.raises(ValueError, match=r"junction 'J2' has a negative surcharge depth"):
        read_swmm_inp(path)
    bad = dataclasses.replace(net, manholes=(
        dataclasses.replace(net.manholes[0], surcharge_depth=-1.0), *net.manholes[1:]))
    with pytest.raises(ValueError, match=r"manhole 'J1' surcharge_depth must be >= 0 m"):
        bad.validate()


def test_a_reoriented_kinwave_pipe_does_not_record_a_reversed_gate(tmp_path):
    """Only dynamic wave models gates; the kinematic reader reorients a pipe written against
    the fall (C1 here, J3 -> J1) but leaves `flap_gate_reversed` at its default."""
    text = (DATA / "tree_kinwave.inp").read_text().replace(
        "C1               J1               J3", "C1               J3               J1")
    path = tmp_path / "k.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    c1 = next(p for p in net.pipes if p.name == "C1")
    assert (c1.u, c1.v) == ("J1", "J3")
    assert not any(p.flap_gate_reversed for p in net.pipes)


J4_LINE = "J4               9.0        5          0          0          0\n"
STORAGE_SECTIONS = (
    "[STORAGE]\n"
    ";;Name Elev MaxDepth InitDepth Shape Curve/Params SurDepth Fevap\n"
    "J4      9.0  3.0      0.2       TABULAR  Bowl  0.5  0\n"
    "J3      10.0 2.0      0         FUNCTIONAL 20 1.5 5\n"
    "\n[CURVES]\n"
    ";;Name Type x y\n"
    "Bowl    STORAGE  0 10  0.5 30\n"
    "Bowl             1.0 60\n"
    "Bowl             2.0 80\n"
    "\n[OUTFALLS]\n"
)
J3_LINE = "J3               10.0       5          0          0          0\n"


def _storage(tmp_path, edits=()):
    return _dyn(tmp_path, [(J4_LINE, ""), (J3_LINE, ""), ("[OUTFALLS]\n", STORAGE_SECTIONS),
                           *edits])


def test_storage_units_are_read_with_their_curves(tmp_path):
    """[STORAGE] TABULAR (a [CURVES] STORAGE curve, its points over several lines) and
    FUNCTIONAL (``a1 a2 a0`` in the file, ``A = a1 y^a2 + a0``, i.e. noodl's ("functional",
    a, b, c) = (a1, a2, a0)); MaxDepth sets the ground, InitDepth and SurDepth are read
    (node.c:654-809); absent SurDepth and Fevap are 0. Storage units are manholes, so
    [INFLOWS] reaches them."""
    net, _, _ = read_swmm_inp(_storage(tmp_path))
    units = {m.name: m for m in net.manholes if m.storage is not None}
    assert set(units) == {"J4", "J3"}
    j4, j3 = units["J4"], units["J3"]
    assert j4.storage == ("tabular", (0.0, 0.5, 1.0, 2.0), (10.0, 30.0, 60.0, 80.0))
    assert (j4.invert, j4.ground, j4.initial_depth, j4.surcharge_depth) == (9.0, 12.0, 0.2,
                                                                           0.5)
    assert j3.storage == ("functional", 20.0, 1.5, 5.0)
    assert (j3.ground, j3.initial_depth, j3.surcharge_depth) == (12.0, 0.0, 0.0)
    assert {m.name for m in net.manholes if m.storage is None} == {"J1", "J2", "J5"}


@pytest.mark.parametrize("shape,params,expected", [
    ("CYLINDRICAL", "4 2 0", ("cylindrical", 4.0, 2.0)),
    ("CONICAL", "4 2 0.5", ("conical", 4.0, 2.0, 0.5)),
    ("PARABOLOID", "4 2 3", ("paraboloid", 4.0, 2.0, 3.0)),
    ("PARABOLIC", "4 2 3", ("paraboloid", 4.0, 2.0, 3.0)),      # SWMM 5.2.4's own keyword
    ("PYRAMIDAL", "4 2 0.5", ("pyramidal", 4.0, 2.0, 0.5)),
])
def test_closed_storage_shapes_are_read(tmp_path, shape, params, expected):
    """SWMM 5.2's CYLINDRICAL (L W, a third, unused value), CONICAL (L W Z), PARABOLOID
    (L W H) and PYRAMIDAL (L W Z) (node.c:727-783)."""
    path = _storage(tmp_path, [("FUNCTIONAL 20 1.5 5", f"{shape} {params}")])
    net, _, _ = read_swmm_inp(path)
    assert next(m for m in net.manholes if m.name == "J3").storage == expected


@pytest.mark.parametrize("edit,match", [
    (("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5 5 0 0 1.0 0.1 0.2 0.3"), "exfiltration"),
    (("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5 5 0 0 -1.0 0.5 0.2"), "exfiltration needs"),
    (("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5 5 0 0 1.0 0.5 1.5"), "exfiltration needs"),
    (("FUNCTIONAL 20 1.5 5", "PARABOLIC 2 1 3 0 0 0.5"), "PARABOLOID with an exfiltration"),
    (("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5"), "FUNCTIONAL needs"),
    (("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5 -5"), r"c \(area at zero depth"),
    (("FUNCTIONAL 20 1.5 5", "BOWL 1 2 3"), "shape"),
    (("TABULAR  Bowl", "TABULAR  Cup"), "curve 'Cup'"),
    (("Bowl    STORAGE", "Bowl    PUMP1"), "not a STORAGE curve"),
    (("Bowl             1.0 60", "Bowl             0.4 60"), "strictly increasing"),
    (("J3      10.0 2.0      0", "J3      10.0 -2.0     0"), "negative maximum depth"),
    (("J3      10.0 2.0      0", "J3      10.0 2.0      -1"), "negative initial depth"),
    (("0.5  0\n", "-0.5  0\n"), "negative surcharge depth"),
])
def test_storage_refusals_name_the_unit_and_the_line(tmp_path, edit, match):
    """A malformed exfiltration (the Green-Ampt fields after Fevap) is refused, and an
    exfiltrating PARABOLOID (SWMM leaves its areas unset); so are SWMM's own input errors
    (node.c:705-738, table.c:285-329) and a curve of another type (no other object that
    uses one is read)."""
    with pytest.raises(ValueError, match=rf"line \d+: .*{match}"):
        read_swmm_inp(_storage(tmp_path, [edit]))


def test_storage_and_curves_are_dynamic_wave_only(tmp_path):
    """Under KINWAVE a storage unit is a different model (its outflow from its volume,
    flowrout.c `storage_getOutflow`), so it is refused by name, not read as a junction."""
    path = _storage(tmp_path, [("FLOW_ROUTING         DYNWAVE", "FLOW_ROUTING         KINWAVE")])
    with pytest.raises(ValueError, match=r"\[STORAGE\].*dynamic-wave"):
        read_swmm_inp(path)


def test_a_curve_section_without_storage_units_is_refused_unless_storage(tmp_path):
    """[CURVES] is read for the types an object uses: a WEIR (coefficient) curve drives nothing
    that is modelled."""
    path = _dyn(tmp_path, [("[OUTFALLS]\n", "[CURVES]\nP1 WEIR 0 0 1 1\n\n[OUTFALLS]\n")])
    with pytest.raises(ValueError, match="P1.*type WEIR"):
        read_swmm_inp(path)


def test_storage_evaporation_and_exfiltration_are_read(tmp_path):
    """Fevap and, after it, exfil.c:48-66's Ksat alone, or suction head, Ksat and IMDmax (in or
    mm, in/hr or mm/hr; SI here): a Ksat of 0 creates no exfiltration in either form; two
    values are SWMM's own input error. The climate is attached (`SewerNetwork.climate`)."""
    for params in ("0 0 0", "0", "1.5 0 0.3"):
        path = _storage(tmp_path, [("FUNCTIONAL 20 1.5 5", f"FUNCTIONAL 20 1.5 5 0 0 {params}")])
        net, _, _ = read_swmm_inp(path)
        assert all(m.exfiltration is None for m in net.manholes)
    path = _storage(tmp_path, [("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5 5 0.5 1.5 36")])
    net, _, _ = read_swmm_inp(path)
    j3 = next(m for m in net.manholes if m.name == "J3")
    assert j3.evaporation_factor == 1.5
    assert j3.exfiltration == Exfiltration(conductivity=1e-5, us_units=False)    # 36 mm/hr
    assert net.climate is not None and net.hydrology is None
    path = _storage(tmp_path, [("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5 5 0 0 60 0.25 0.3")])
    j3 = next(m for m in read_swmm_inp(path)[0].manholes if m.name == "J3")
    assert j3.exfiltration == Exfiltration(conductivity=0.25 / 3.6e6, suction=0.06,
                                           initial_deficit=0.3, us_units=False)
    path = _storage(tmp_path, [("FUNCTIONAL 20 1.5 5", "FUNCTIONAL 20 1.5 5 0 0 1.5 0.2")])
    with pytest.raises(ValueError, match=r"line \d+: storage unit 'J3'.*2 exfiltration values"):
        read_swmm_inp(path)


STRUCTURES = "structures_dynwave.inp"


def _structures(tmp_path, edits=()):
    text = (DATA / STRUCTURES).read_text()
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / "regulators.inp"
    path.write_text(text)
    return path


def test_orifices_weirs_and_outlets_are_read():
    """[ORIFICES], [WEIRS] and [OUTLETS] (link.c:1641-1692, 2010-2099, 2524-2604) with
    their [XSECTIONS] openings and the [CURVES] RATING curve of a tabular outlet."""
    import dataclasses

    from noodl.apps.sewer.network import Orifice, Outlet, Weir
    net, _, _ = read_swmm_inp(DATA / STRUCTURES)
    for kind in ("orifices", "weirs", "outlets"):     # `order` is the line of the entry
        assert all(d.order > 0 for d in getattr(net, kind))
        net = dataclasses.replace(net, **{kind: tuple(dataclasses.replace(d, order=0)
                                                      for d in getattr(net, kind))})
    assert net.orifices == (Orifice("OR1", "J2", "J3", "side", "rectangular", 0.2, 0.1, 0.65,
                                    width=0.3),)
    assert net.weirs == (
        Weir("W1", "J4", "J5", "transverse", 1.0, 0.5, 0.3, 1.84, end_contractions=2.0),
        Weir("W2", "J7", "J4", "v_notch", 0.6, 1.2, 0.1, 1.38))
    assert net.outlets == (Outlet("OL1", "J8", "Out2", 0.05,
                                  ("tabular", (0.0, 0.2, 0.5, 1.0), (0.0, 0.01, 0.03, 0.05))),)


def test_regulator_options_and_defaults_are_read(tmp_path):
    """A weir's ``*`` keeps a default (link.c:2043-2066); a gated orifice, a weir that cannot
    surcharge, a trapezoidal weir's side slopes, a circular bottom orifice, FUNCTIONAL/HEAD
    outlets with a gate, and a negative outlet offset read as 0 (link.c:2556)."""
    path = _structures(tmp_path, [
        ("OR1              J2               J3               SIDE         0.1        0.65"
         "       NO       0",
         "OR1 J2 J3 BOTTOM 0.1 0.65 YES 0.5"),
        ("OR1              RECT_CLOSED  0.2              0.3        0          0",
         "OR1 CIRCULAR 0.25 0 0 0"),
        ("W1               J4               J5               TRANSVERSE   0.3        1.84"
         "       NO       2        0          YES",
         "W1 J4 J5 TRAPEZOIDAL 0.3 1.84 * * 1.2 NO"),
        ("W1               RECT_OPEN    1.0              0.5        0          0",
         "W1 TRAPEZOIDAL 1.0 0.5 0.5 1.5"),
        ("OL1              J8               Out2             0.05       TABULAR/DEPTH"
         "   Rate1                       NO",
         "OL1 J8 Out2 -0.2 FUNCTIONAL/HEAD 0.3 0.6 YES"),
    ])
    net, _, _ = read_swmm_inp(path)
    orifice = net.orifices[0]
    assert (orifice.orientation, orifice.shape, orifice.height, orifice.width,
            orifice.flap_gate) == ("bottom", "circular", 0.25, None, True)
    w1 = net.weirs[0]
    assert (w1.kind, w1.side_slopes, w1.end_contractions, w1.end_discharge_coefficient,
            w1.can_surcharge, w1.flap_gate) == ("trapezoidal", (0.5, 1.5), 0.0, 1.2, False,
                                                False)
    outlet = net.outlets[0]
    assert (outlet.crest_height, outlet.rating, outlet.head_difference, outlet.flap_gate) == (
        0.0, ("functional", 0.3, 0.6), True, True)


ORIFICE_LINE = ("OR1              J2               J3               SIDE         0.1        0.65"
                "       NO       0")
W1_LINE = ("W1               J4               J5               TRANSVERSE   0.3        1.84"
           "       NO       2        0          YES")
OL1_LINE = ("OL1              J8               Out2             0.05       TABULAR/DEPTH"
            "   Rate1                       NO")


@pytest.mark.parametrize("edit,match", [
    ((W1_LINE, "W1 J4 J5 ROADWAY 0.3 1.84"), "ROADWAY weirs are not modelled"),
    ((W1_LINE, W1_LINE + " 0 PAVED WeirCd"), "coefficient curve 'WeirCd'"),
    ((W1_LINE, "W1 J4 J5 LATERAL 0.3 1.84"), "type must be one of"),
    ((W1_LINE, "W1 J4 J5 TRANSVERSE -0.3 1.84"), "negative crest height"),
    ((W1_LINE, "W1 J4 J5 TRANSVERSE 0.3 -1.84"), "negative discharge coefficient"),
    ((ORIFICE_LINE, "OR1 J2 J3 TOP 0.1 0.65"), "SIDE or BOTTOM"),
    ((ORIFICE_LINE, "OR1 J2 J3 SIDE 0.1 0.65 MAYBE"), "flap gate must be YES or NO"),
    ((ORIFICE_LINE, "OR1 J2 J9 SIDE 0.1 0.65"), "names node 'J9'"),
    (("OR1              RECT_CLOSED", "OR1              RECT_OPEN  "), "ERR_REGULATOR_SHAPE"),
    (("W2               TRIANGULAR ", "W2               RECT_OPEN  "), "ERR_REGULATOR_SHAPE"),
    (("W2               TRIANGULAR   0.6              1.2        0          0\n", ""),
     "'W2' has no \\[XSECTIONS\\]"),
    (("W2               TRIANGULAR   0.6              1.2        0          0\n",
      "W2 TRIANGULAR 0.6 1.2 0 0\nOL1 CIRCULAR 0.3 0 0 0\n"), "outlet 'OL1' has an"),
    ((OL1_LINE, "OL1 J8 Out2 0.05 TABULAR/DEPTH Missing"), "not a RATING curve"),
    ((OL1_LINE, "OL1 J8 Out2 0.05 CONICAL Rate1"), "TABULAR/DEPTH"),
    (("Rate1            Rating", "Rate1            Weir  "), "type WEIR"),
    (("LINK_OFFSETS         DEPTH", "LINK_OFFSETS         ELEVATION"),
     "LINK_OFFSETS DEPTH only"),
])
def test_regulator_refusals_name_the_device(tmp_path, edit, match):
    """Roadway weirs, weir coefficient curves, culvert-like extras and anything SWMM itself
    rejects (ERR_REGULATOR_SHAPE, ERR_KEYWORD, ERR_NAME) are refused by name with the
    line; so are regulators under STEADY/KINWAVE and with LINK_OFFSETS ELEVATION."""
    with pytest.raises(ValueError, match=match):
        read_swmm_inp(_structures(tmp_path, [edit]))


def test_regulators_are_dynamic_wave_only(tmp_path):
    """Under KINWAVE SWMM routes regulators by their inflow alone (link.c: ``h2 =
    node1's invert``), a different model, so the sections are refused by name."""
    text = (DATA / "tree_kinwave.inp").read_text() + "\n[ORIFICES]\nOR1 J1 J3 SIDE 0.1 0.65\n"
    path = tmp_path / "k.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"\[ORIFICES\].*flow regulators are read under "
                                         r"dynamic-wave routing only"):
        read_swmm_inp(path)


# ----------------------------------------------------------------------------- dividers
def _divider_inp(tmp_path, row, dynwave=True, extra=()):
    """The DYNWAVE tree with J3 moved from [JUNCTIONS] to [DIVIDERS] as `row`."""
    path = _dyn(tmp_path) if dynwave else DATA / "tree_kinwave.inp"
    text = path.read_text()
    j3 = next(line for line in text.splitlines() if line.startswith("J3 "))
    text = text.replace(j3 + "\n", "").replace(
        "[OUTFALLS]", f"[DIVIDERS]\n{row}\n\n[OUTFALLS]", 1)
    for old, new in extra:
        assert old in text, old
        text = text.replace(old, new)
    out = tmp_path / "divider.inp"
    out.write_text(text)
    return out


@pytest.mark.parametrize("row,kind", [
    ("J3 10.0 C3 OVERFLOW 0.7 0.1 0.2 5", "overflow"),
    ("J3 10.0 C3 CUTOFF 0.05 0.7 0.1 0.2 5", "cutoff"),
    ("J3 10.0 C3 TABULAR D1 0.7 0.1 0.2 5", "tabular"),
    ("J3 10.0 C3 WEIR 0.01 0.5 3.0 0.7 0.1 0.2 5", "weir"),
])
def test_a_divider_is_read_as_a_junction_with_its_depths(tmp_path, row, kind):
    """node.c:1157-1208: the MaxDepth, InitDepth, SurDepth and Aponded follow the fields
    of the divider type (none for OVERFLOW, one for CUTOFF and TABULAR, three for WEIR);
    under DYNWAVE the divider is a plain junction (the diversion is never called)."""
    text = _divider_inp(tmp_path, row).read_text()
    path = tmp_path / "read.inp"
    path.write_text(text + "\n[CURVES]\nD1 DIVERSION 0 0 1 1\n")
    net, _, _ = read_swmm_inp(path)
    j3 = next(m for m in net.manholes if m.name == "J3")
    assert (j3.invert, j3.ground, j3.initial_depth, j3.surcharge_depth) == (
        10.0, pytest.approx(10.7), 0.1, 0.2)


@pytest.mark.parametrize("row,match", [
    ("J3 10.0 C9 OVERFLOW", "'C9', which is not a link"),
    ("J3 10.0 C4 OVERFLOW", "'C4', which is not attached"),
    ("J3 10.0 * OVERFLOW", "no diverted link"),
    ("J3 10.0 C3 SPILL", "'SPILL'"),
    ("J3 10.0 C3 CUTOFF", "needs 5 fields"),
    ("J3 10.0 C3 TABULAR NoCurve", "'NoCurve', which is not in"),
    ("J3 10.0 C3 CUTOFF abc", "must be a number"),
    ("J3 10.0 C3 WEIR 0.01 0 3.0", "ERR_WEIR_DIVIDER"),
    ("J3 10.0 C3 WEIR 0.01 0.5 0", "ERR_WEIR_DIVIDER"),
    ("J3 10.0 C3 WEIR 9 0.5 3.0", "ERR_WEIR_DIVIDER"),
    ("J3 10.0 C3 OVERFLOW 5 0 -1", "negative surcharge depth"),
])
def test_divider_refusals_are_swmms_own(tmp_path, row, match):
    """What `node_validate`/`divider_readParams` reject whatever the routing."""
    with pytest.raises(ValueError, match=match):
        read_swmm_inp(_divider_inp(tmp_path, row))


def test_dividers_are_refused_by_name_under_kinwave_and_steady(tmp_path):
    """Outside DYNWAVE SWMM diverts flow at a divider (node.c `divider_getOutflow`), which
    is not modelled."""
    with pytest.raises(ValueError, match=r"\[DIVIDERS\].*dynamic-wave routing only"):
        read_swmm_inp(_divider_inp(tmp_path, "J3 10.0 C3 OVERFLOW", dynwave=False))


# ------------------------------------------------------------- content before a section
def test_lines_before_the_first_section_are_title_text_as_in_swmm(tmp_path):
    """SWMM's second pass over the file starts in `[TITLE]` (input.c:174), so text before
    the first header is title text; the model is the file's own."""
    path = tmp_path / "title.inp"
    path.write_text("A stray title line\n; a comment\nJ1 12.0 not a record\n"
                    + _dyn(tmp_path).read_text())
    assert read_swmm_inp(path)[0].routing == "dynamic_wave"


def test_a_merge_conflict_marker_is_refused(tmp_path):
    """A file with both sides of a merge holds its sections twice: refused, naming the
    marker and its line, wherever it sits."""
    path = tmp_path / "merge.inp"
    path.write_text("<<<<<<< HEAD\n" + _dyn(tmp_path).read_text())
    with pytest.raises(ValueError, match=r"line 1: version-control merge-conflict marker"):
        read_swmm_inp(path)
    path.write_text(_dyn(tmp_path).read_text() + "\n[MAP]\n=======\n")
    with pytest.raises(ValueError, match="merge-conflict marker '======='"):
        read_swmm_inp(path)


# --------------------------------------------------------------------------- ponding
def test_allow_ponding_yes_is_read_with_the_junction_ponded_areas(tmp_path):
    """`ALLOW_PONDING YES` (project.c:603-616, NoYesWords) is `allow_ponding`; Aponded (the
    sixth [JUNCTIONS] field, or the one after SurDepth of a divider) is `ponded_area`, m2."""
    j1 = "J1               12.0       5          0          0          0"
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [
        ("ALLOW_PONDING        NO", "ALLOW_PONDING        YES"),
        (j1, "J1               12.0       5          0          0          2.5")]))
    assert net.dynamic_wave.allow_ponding
    assert {m.name: m.ponded_area for m in net.manholes}["J1"] == 2.5
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [(j1, j1[:-1] + "2.5")]))
    assert not net.dynamic_wave.allow_ponding            # the default is NO
    assert {m.name: m.ponded_area for m in net.manholes}["J1"] == 2.5   # read, unused
    row = "J3 10.0 C3 CUTOFF 0.05 0.7 0.1 0.2 5"
    assert next(m for m in read_swmm_inp(_divider_inp(tmp_path, row))[0].manholes
                if m.name == "J3").ponded_area == 5.0


@pytest.mark.parametrize("old,new,match", [
    ("ALLOW_PONDING        NO", "ALLOW_PONDING        MAYBE", r"line \d+: ALLOW_PONDING must be"),
    ("J1               12.0       5          0          0          0",
     "J1               12.0       5          0          0          -1", "negative ponded area"),
])
def test_ponding_refusals(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read_swmm_inp(_dyn(tmp_path, [(old, new)]))


def test_ponding_with_a_ponded_area_is_dynamic_wave_only(tmp_path):
    """Under STEADY or KINWAVE SWMM re-releases ponded water into the next step's flow
    (flowrout.c:151-156), which is not modelled; without an Aponded it changes nothing."""
    text = (DATA / "tree_kinwave.inp").read_text().replace(
        "ALLOW_PONDING        NO", "ALLOW_PONDING        YES")
    path = tmp_path / "k.inp"
    path.write_text(text)
    read_swmm_inp(path)
    j1 = "J1               12.0       5          0          0          0"
    assert j1 in text
    path.write_text(text.replace(j1, j1[:-1] + "2"))
    with pytest.raises(ValueError, match=r"line \d+: ALLOW_PONDING YES with node 'J1'"):
        read_swmm_inp(path)


# ------------------------------------------------------- options and sections that matter
@pytest.mark.parametrize("old,new,match", [
    ("SKIP_STEADY_STATE    NO", "SKIP_STEADY_STATE    YES", "SKIP_STEADY_STATE YES"),
    ("SKIP_STEADY_STATE    NO", "SKIP_STEADY_STATE    MAYBE", "must be YES or NO"),
    ("SKIP_STEADY_STATE    NO", "SKIP_STEADY_STATE    NO\nIGNORE_ROUTING YES",
     "IGNORE_ROUTING YES"),
    ("FLOW_ROUTING         DYNWAVE", "FLOW_ROUTING         NONE", "FLOW_ROUTING NONE"),
])
@pytest.mark.parametrize("dynwave", [True, False])
def test_options_that_change_what_is_routed_are_refused_by_name(tmp_path, old, new, match,
                                                                dynwave):
    """IGNORE_ROUTING, FLOW_ROUTING NONE and SKIP_STEADY_STATE YES change the routed result
    (swmm5.c:375, routing.c:386-392), under every routing; NO is SWMM's default."""
    if dynwave:
        path = _dyn(tmp_path, [(old, new)] if old != "FLOW_ROUTING         DYNWAVE" else [])
        if old.startswith("FLOW_ROUTING"):
            path.write_text(path.read_text().replace(old, new))
    else:
        old = old.replace("DYNWAVE", "KINWAVE")
        path = tmp_path / "k.inp"
        text = (DATA / "tree_kinwave.inp").read_text()
        assert old in text
        path.write_text(text.replace(old, new.replace("DYNWAVE", "KINWAVE")))
    with pytest.raises(ValueError, match=match):
        read_swmm_inp(path)


OPEN_C1 = [("C1               CIRCULAR     0.30             0          0          0          1",
            "C1               RECT_OPEN    0.30             1.0        0          0          1")]


def test_evaporation_is_read_for_open_conduits_and_refused_where_it_is_not_modelled(tmp_path):
    """An open conduit evaporates (`losses.py`), so the file's `[EVAPORATION]` is read for it:
    CONSTANT, MONTHLY and TIMESERIES become `SewerNetwork.climate`; a TEMPERATURE or FILE
    source is refused naming the line, and only when an open conduit (or a storage unit with an
    evaporation factor, or a seeping conduit) reads it. A closed network ignores the section."""
    net, _, _ = read_swmm_inp(_dyn(tmp_path, OPEN_C1))
    assert net.climate is not None and net.climate.evaporation.kind == "constant"
    assert net.climate_source is net.climate and net.has_wet_losses is False  # rate 0
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [*OPEN_C1, ("CONSTANT         0.0",
                                                         "CONSTANT         0.1")]))
    assert net.has_wet_losses and net.climate.evaporation.monthly == (0.1,) * 12
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [*OPEN_C1, ("CONSTANT         0.0",
                                                         "MONTHLY 0 0 0 0 0 0 0 0 0 0 0 0.2")]))
    assert net.climate.evaporation.kind == "monthly" and net.has_wet_losses
    for row in ("TEMPERATURE", "FILE x.txt"):
        path = _dyn(tmp_path, [*OPEN_C1, ("CONSTANT         0.0", row)])
        with pytest.raises(ValueError, match=r"line \d+: \[EVAPORATION\] .*climate file"):
            read_swmm_inp(path)
        net, _, _ = read_swmm_inp(_dyn(tmp_path, [("CONSTANT         0.0", row)]))   # closed
        assert net.climate is None
    net, _, _ = read_swmm_inp(_dyn(tmp_path, [("CONSTANT         0.0", "CONSTANT         0.1")]))
    assert net.climate is None and not net.has_wet_losses


def test_adjustments_without_subcatchments_are_read_under_dynamic_wave_only(tmp_path):
    """`[ADJUSTMENTS]` feeds the conduit and storage losses (EVAPORATION and CONDUCTIVITY); a
    line that names a subcatchment is refused, and the section stays refused outside DYNWAVE."""
    adj = "\n[ADJUSTMENTS]\nEVAPORATION " + " ".join(["0.1"] * 12) + "\nCONDUCTIVITY " + \
          " ".join(["0.5"] * 12) + "\n"
    path = _dyn(tmp_path, OPEN_C1)
    path.write_text(path.read_text() + adj)
    net, _, _ = read_swmm_inp(path)
    assert net.climate.adjustments.evaporation == (0.1,) * 12
    assert net.climate.adjustments.conductivity == (0.5,) * 12
    assert net.has_wet_losses
    path.write_text(path.read_text() + "N-PERV S1 Pat\n")
    with pytest.raises(ValueError, match="not a subcatchment"):
        read_swmm_inp(path)
    kin = tmp_path / "k.inp"
    kin.write_text((DATA / "tree_kinwave.inp").read_text() + adj)
    with pytest.raises(ValueError, match=r"\[ADJUSTMENTS\].*not part of the documented subset"):
        read_swmm_inp(kin)


def test_seepage_is_read_in_dynamic_wave_and_refused_otherwise(tmp_path):
    """The sixth `[LOSSES]` field is a rate in mm/hr here (SI), stored in m/s; a negative one
    does nothing in SWMM (link.c:1366) and is stored as none; under KINWAVE a positive one is
    refused by name."""
    path = _dyn(tmp_path, [("[INFLOWS]", "[LOSSES]\nC1  0  0  0  NO  36\nC2 0 0 0 NO -5\n\n"
                                           "[INFLOWS]")])
    net, _, _ = read_swmm_inp(path)
    seep = {p.name: p.seepage for p in net.pipes}
    assert seep["C1"] == pytest.approx(1e-5) and seep["C2"] == 0.0
    assert net.has_wet_losses and net.climate is not None
    kin = tmp_path / "k.inp"
    kin.write_text((DATA / "tree_kinwave.inp").read_text() + "\n[LOSSES]\nC1 0 0 0 NO 3.0\n")
    with pytest.raises(ValueError, match="seepage rate 3.0; conduit seepage is modelled by dyn"):
        read_swmm_inp(kin)


def test_an_inflow_at_an_outfall_reaches_no_link_and_is_dropped(tmp_path):
    """A lateral inflow into an outfall (a node dynamic-wave routing never solves) is read and
    dropped; an unknown node is still refused."""
    inflows = "J1               FLOW             \"\"               FLOW     1.0      1.0      0.05"
    edit = [(inflows, inflows + "\nOutfall          FLOW             \"\"               FLOW"
             "     1.0      1.0      0.07")]
    net, _, _ = read_swmm_inp(_dyn(tmp_path, edit))
    base, _, _ = read_swmm_inp(_dyn(tmp_path))
    assert net.manholes == base.manholes and net.outfalls == base.outfalls
    edit = [(inflows, inflows + "\nNowhere          FLOW             \"\"               FLOW"
             "     1.0      1.0      0.07")]
    with pytest.raises(ValueError, match="inflow at 'Nowhere', which is not a junction"):
        read_swmm_inp(_dyn(tmp_path, edit))


def test_raingages_without_subcatchments_are_ignored(tmp_path):
    """A gage only feeds subcatchments (and RDII, a refused section of its own)."""
    gages = ("\n[RAINGAGES]\n;;Name Format Interval SCF Source\nG1 VOLUME 0:05 1.0 "
             "TIMESERIES R1\n")
    path = tmp_path / "gage.inp"
    path.write_text(_dyn(tmp_path).read_text() + gages)
    read_swmm_inp(path)
    path.write_text(path.read_text() + "\n[SUBCATCHMENTS]\nS1 G1 J1 1 50 100 0.5 0\n")
    with pytest.raises(ValueError, match="time series 'R1', which has no"):
        read_swmm_inp(path)         # subcatchments are read now (tests/apps/sewer/test_runoff.py)


NEG = ("C1               J1               J3               200        0.013      0"
       "          0")


@pytest.mark.parametrize("dynwave", [False, True])
@pytest.mark.parametrize("offsets", ["DEPTH", "ELEVATION"])
def test_a_negative_conduit_offset_is_zero(tmp_path, dynwave, offsets):
    """link.c:1057-1067 (`conduit_validate`, DEPTH) and :498-502 (`link_getOffsetHeight`,
    ELEVATION) set a negative offset height to 0 with a warning; the slope is taken from the
    clamped end elevations (link.c:1269), so C1 (inlet 0.3 m below J1's invert) is the same
    conduit as with no offsets."""
    text = (DATA / "tree_kinwave.inp").read_text()
    if dynwave:
        text = (text.replace("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE")
                .replace("VARIABLE_STEP        0.75", "VARIABLE_STEP        0"))
    assert text.count(NEG) == 1
    c1 = NEG[:-18] + ("      -0.3        0" if offsets == "DEPTH" else "      11.7       10")
    text = text.replace(NEG, c1).replace("LINK_OFFSETS         DEPTH",
                                         f"LINK_OFFSETS         {offsets}")
    (tmp_path / "neg.inp").write_text(text)
    net, _, _ = read_swmm_inp(tmp_path / "neg.inp")
    net.validate()
    pipe = next(p for p in net.pipes if p.name == "C1")
    assert (pipe.inlet_offset, pipe.outlet_offset) == (0.0, 0.0)
    assert pipe.slope == pytest.approx(2.0 / (200.0**2 - 2.0**2) ** 0.5, rel=1e-12)


def test_a_junction_with_no_link_and_an_outfall_free_group_are_allowed_under_dynwave(tmp_path):
    """SWMM asks for one outfall in the network (flowrout.c:325) and nothing per component:
    the dynamic-wave reader accepts a junction no link reaches and a conduit pair draining
    nowhere; KINWAVE keeps the tree rules, by name."""
    from noodl.apps.sewer import build_model
    path = _dyn(tmp_path, [
        ("[JUNCTIONS]\n", "[JUNCTIONS]\nJX 5.0 2 0 0 0\nJA 5.0 2 0 0 0\nJB 4.0 2 0 0 0\n"),
        ("[XSECTIONS]\n", "[XSECTIONS]\nCX CIRCULAR 0.30 0 0 0 1\n"),
        ("[CONDUITS]\n", "[CONDUITS]\nCX JA JB 50 0.013 0 0 0\n")])
    net, _, _ = read_swmm_inp(path)
    net.validate()
    model, _, _ = build_model(net)
    assert "JX" in model.dynamic_wave.junction_names
    kin = tmp_path / "k.inp"
    kin.write_text(path.read_text().replace("FLOW_ROUTING         DYNWAVE",
                                            "FLOW_ROUTING         KINWAVE"))
    with pytest.raises(ValueError, match="'JX' has 0 outgoing"):
        read_swmm_inp(kin)[0].validate()


def test_a_dynwave_network_without_any_outfall_is_refused():
    from noodl.apps.sewer.network import Manhole, Pipe, SewerNetwork
    net = SewerNetwork((Manhole("A", 1.0), Manhole("B", 0.5)),
                       (Pipe("AB", "A", "B", 100.0, 0.3, 0.013, 0.005),), (),
                       routing="dynamic_wave")
    with pytest.raises(ValueError, match="no outfall"):
        net.validate()
