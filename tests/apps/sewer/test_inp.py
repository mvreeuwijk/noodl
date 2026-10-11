"""The documented SWMM .inp subset."""

import dataclasses
import math
from pathlib import Path

import pytest

from noodl.apps.sewer.inp import read_swmm_inp

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


def test_non_cms_flow_units_are_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "FLOW_UNITS           CMS", "FLOW_UNITS           CFS"
    )
    path = tmp_path / "cfs.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="FLOW_UNITS must be CMS"):
        read_swmm_inp(path)


def test_an_unknown_section_with_content_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text() + "\n[PUMPS]\nP1 J1 J3 HEAD C1\n"
    path = tmp_path / "pumps.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"\[PUMPS\]"):
        read_swmm_inp(path)


def test_a_time_series_inflow_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        'J1               FLOW             ""               FLOW     1.0      1.0      0.05',
        "J1               FLOW             TS1              FLOW     1.0      1.0      0.05",
    )
    path = tmp_path / "ts.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="time series"):
        read_swmm_inp(path)


def test_a_non_circular_cross_section_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "C1               CIRCULAR", "C1               RECT_CLOSED"
    )
    path = tmp_path / "rect.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="only CIRCULAR"):
        read_swmm_inp(path)


def test_a_conduit_with_zero_fall_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "J3               10.0       5", "J3               12.0       5"
    )
    path = tmp_path / "flat.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="zero or adverse fall"):
        read_swmm_inp(path)


def test_a_dwf_section_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text() + "\n[DWF]\nJ1 FLOW 0.01\n"
    path = tmp_path / "dwf.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"\[DWF\]"):
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
    ("LENGTHENING_STEP     0", "LENGTHENING_STEP     10", "LENGTHENING_STEP"),
    ("ALLOW_PONDING        NO", "ALLOW_PONDING        YES", "ALLOW_PONDING"),
    ("J1               12.0       5          0", "J1               12.0       5          0.2",
     "initial depth"),
    ("C1               J1               J3               200        0.013      0          0"
     "          0", "C1               J1               J3               200        0.013"
     "      0          0          0.01", "initial flow"),
    ("C1               CIRCULAR     0.30             0          0          0          1",
     "C1               CIRCULAR     0.30             0          0          0          2",
     "barrels"),
    ("Outfall          8.0        FREE                        NO",
     "Outfall          8.0        FREE                        YES", "flap gate"),
    ("Outfall          8.0        FREE                        NO",
     "Outfall          8.0        TIDAL      Tide1            NO", "TIDAL"),
])
def test_dynwave_phase_a_refusals(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read_swmm_inp(_dyn(tmp_path, [(old, new)]))


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
