"""The documented SWMM .inp subset (spec 4.5)."""

from pathlib import Path

import pytest

from noodl.apps.sewer.inp import read_inp

DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"


def test_the_committed_fixture_round_trips_to_the_research_network():
    net, inflows, pollutants = read_inp(DATA / "tree_steady.inp")
    net.validate()
    assert [m.name for m in net.manholes] == ["J1", "J2", "J5", "J3", "J4"]
    assert [o.name for o in net.outfalls] == ["Outfall"]
    assert [p.name for p in net.pipes] == ["C1", "C2", "C3", "C4", "C5"]
    # 5 decimals, not 10 (deviation from the brief; see the Task 9 report): the reader's own
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
    net, inflows, pollutants = read_inp(DATA / "tree_kinwave_pollut.inp")
    assert set(pollutants) == {"Tracer"}
    # MG/L is converted to kg/m3 by 1e-3, and Kdecay 1/day to 1/s
    assert pollutants["Tracer"]["decay"] == pytest.approx(5.0 / 86400.0, rel=1e-14)
    assert inflows == {"J1": {"Tracer": pytest.approx(0.1, rel=1e-14)}}


def test_conduit_offsets_shift_the_effective_invert_under_link_offsets_depth(tmp_path):
    """M4-R19 minor: under `LINK_OFFSETS DEPTH` (the fixture's own default), a conduit's
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
    net, _, _ = read_inp(path)
    pipe = next(p for p in net.pipes if p.name == "C1")
    fall = 12.0 - (10.0 + 0.5)
    assert pipe.slope == pytest.approx(fall / (200.0**2 - fall**2) ** 0.5, rel=1e-12)


def test_conduits_are_normalised_upstream_to_downstream():
    """C1 is written J1 -> J3 with inverts 12.0 and 10.0, so it already runs downhill; the
    reader still recomputes the slope from the inverts and the 3-D chord length."""
    net, _, _ = read_inp(DATA / "tree_steady.inp")
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
        read_inp(path)


def test_an_unknown_section_with_content_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text() + "\n[PUMPS]\nP1 J1 J3 HEAD C1\n"
    path = tmp_path / "pumps.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"\[PUMPS\]"):
        read_inp(path)


def test_a_time_series_inflow_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        'J1               FLOW             ""               FLOW     1.0      1.0      0.05',
        "J1               FLOW             TS1              FLOW     1.0      1.0      0.05",
    )
    path = tmp_path / "ts.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="time series"):
        read_inp(path)


def test_a_non_circular_cross_section_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "C1               CIRCULAR", "C1               RECT_CLOSED"
    )
    path = tmp_path / "rect.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="only CIRCULAR"):
        read_inp(path)


def test_a_conduit_with_zero_fall_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "J3               10.0       5", "J3               12.0       5"
    )
    path = tmp_path / "flat.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="zero or adverse fall"):
        read_inp(path)


def test_a_dwf_section_is_refused(tmp_path):
    text = (DATA / "tree_steady.inp").read_text() + "\n[DWF]\nJ1 FLOW 0.01\n"
    path = tmp_path / "dwf.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"\[DWF\]"):
        read_inp(path)


def test_the_kinwave_variant_records_its_routing():
    net, _, _ = read_inp(DATA / "tree_kinwave.inp")
    assert net.routing == "KINWAVE"
