"""The MATPOWER v2 case-file reader."""

import math

import pytest

from noodl.apps.electrical.power import read_matpower
from tests._power_cases import power_case


@pytest.mark.external
def test_case9_tables_in_file_units():
    net = read_matpower(power_case("case9"))
    assert net.name == "case9" and net.base_mva == 100.0
    assert net.bus_ids() == tuple(range(1, 10))
    assert [b.kind for b in net.buses[:4]] == ["slack", "pv", "pv", "pq"]
    assert [(ld.bus, ld.p_mw, ld.q_mvar) for ld in net.loads] == [
        (5, 90.0, 30.0), (7, 100.0, 35.0), (9, 125.0, 50.0)]
    assert net.shunts == ()
    g = net.generators[0]
    assert (g.bus, g.p_mw, g.vm_set, g.in_service) == (1, 72.3, 1.04, True)
    br = net.branches[1]
    assert (br.from_bus, br.to_bus, br.r, br.x, br.b, br.ratio, br.shift_deg) == (
        4, 5, 0.017, 0.092, 0.158, 0.0, 0.0)
    assert net.notes["ignored_fields"] == ["gencost"]


@pytest.mark.external
def test_case118_ignores_names_and_keeps_its_slack_angle():
    net = read_matpower(power_case("case118"))
    assert len(net.buses) == 118 and len(net.branches) == 186 and len(net.generators) == 54
    slack = [b for b in net.buses if b.kind == "slack"]
    assert [(b.id, b.va_deg) for b in slack] == [(69, 30.0)]
    assert net.notes["ignored_fields"] == ["gencost", "bus_name"]


@pytest.mark.external
def test_case14_shunt_and_taps():
    net = read_matpower(power_case("case14"))
    assert [(s.bus, s.g_mw, s.b_mvar) for s in net.shunts] == [(9, 0.0, 19.0)]
    assert sum(1 for br in net.branches if br.ratio != 0) == 3


@pytest.mark.external
def test_features_case_keeps_out_of_service_elements():
    net = read_matpower(power_case("case9_features"))
    assert [g.in_service for g in net.generators] == [True, True, True, False, True]
    assert [br.in_service for br in net.branches][-1] is False
    assert (net.branches[0].ratio, net.branches[0].shift_deg) == (1.02, -3.0)
    assert [(s.bus, s.g_mw, s.b_mvar) for s in net.shunts] == [(5, 5.0, 10.0)]
    assert math.isclose(net.buses[0].va_deg, 10.0)


def _case(tmp_path, body, name="tiny"):
    path = tmp_path / f"{name}.m"
    path.write_text(
        f"function mpc = {name}\nmpc.version = '2';\nmpc.baseMVA = 100;\n"
        "mpc.bus = [\n 1 3 0 0 0 0 1 1 0 1 1 1.1 0.9;\n 2 1 50 10 0 0 1 1 0 1 1 1.1 0.9;\n];\n"
        "mpc.gen = [\n 1 0 0 300 -300 1 100 1 250 10;\n];\n"
        "mpc.branch = [\n 1 2 0.01 0.1 0 0 0 0 0 0 1 -360 360;\n];\n" + body,
        encoding="utf-8")
    return path


def test_a_minimal_hand_written_case_reads(tmp_path):
    net = read_matpower(_case(tmp_path, ""))
    assert net.name == "tiny" and [ld.p_mw for ld in net.loads] == [50.0]


@pytest.mark.parametrize("body, match", [
    ("mpc.bus(:, 8) = 1;\n", r"line 14: 'mpc.bus\(:, 8\) = 1;'"),
    ("define_constants;\n", "define_constants"),
    ("mpc.dcline = [\n 1 2 1 10 10 0 0 1 1 0 100 -100 100 -100 100 0 0 ];\n", "DC lines"),
    ("mpc.foo = [1];\n", "mpc.foo"),
])
def test_statements_outside_the_subset_are_refused(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        read_matpower(_case(tmp_path, body))


@pytest.mark.parametrize("old, new, match", [
    (" 2 1 50", " 2 4 50", r"isolated.*\[2\]"),
    ("'2'", "'1'", "version 2"),
    (" 1 2 0.01 0.1 0 0 0 0 0 0 1 -360 360;", " 1 2 0.01;", r"branch.*11 columns"),
    (" 1 2 0.01 0.1", " 1 2 0.01 abc", r"line 12.*'abc'"),
])
def test_isolated_bus_version_short_rows_and_bad_tokens_are_refused(tmp_path, old, new, match):
    p = _case(tmp_path, "")
    p.write_text(p.read_text().replace(old, new))
    with pytest.raises(ValueError, match=match):
        read_matpower(p)


def test_continuations_commas_inf_and_quoted_percent_signs(tmp_path):
    p = _case(tmp_path, "mpc.bus_name = {\n 'Bus % one';\n 'Two';\n};\n")
    p.write_text(
        p.read_text()
        .replace(" 1 0 0 300 -300 1", " 1, 0, 0, Inf, -Inf, ... reactive limits\n 1")
        .replace("];\nmpc.gen", "] ; % end of the bus table\nmpc.gen"))
    net = read_matpower(p)
    g = net.generators[0]
    assert (g.q_max_mvar, g.q_min_mvar, g.vm_set) == (math.inf, -math.inf, 1.0)
    assert net.notes["ignored_fields"] == ["bus_name"]


@pytest.mark.parametrize("body, match", [
    ("mpc.branch = [\n 1 2 0.01 0.1 0 0 0 0 0 0 1;\n];\n", "mpc.branch.*more than once"),
    ("mpc.gencost = {\n 'a' };\nmpc.areas = [1 1];\n", None),
    ("mpc.bus_name = [1];\nfunction mpc = again\n", "again"),
])
def test_repeated_fields_and_functions_are_refused(tmp_path, body, match):
    if match is None:
        assert read_matpower(_case(tmp_path, body)).notes["ignored_fields"] == [
            "gencost", "areas"]
        return
    with pytest.raises(ValueError, match=match):
        read_matpower(_case(tmp_path, body))


def test_a_missing_table_and_a_cell_array_for_a_table_are_refused(tmp_path):
    p = _case(tmp_path, "")
    p.write_text(p.read_text().replace("mpc.gen = [\n 1 0 0 300 -300 1 100 1 250 10;\n];\n", ""))
    with pytest.raises(ValueError, match="mpc.gen"):
        read_matpower(p)
    p.write_text(_case(tmp_path, "").read_text().replace(
        "mpc.gen = [\n 1 0 0 300 -300 1 100 1 250 10;\n];",
        "mpc.gen = {\n 1 0 0 300 -300 1 100 1 250 10;\n};"))
    with pytest.raises(ValueError, match=r"line 8.*mpc.gen.*numeric matrix"):
        read_matpower(p)


@pytest.mark.external
@pytest.mark.parametrize("case, sizes, ignored", [
    ("case9", (9, 9, 3), ["gencost"]),
    ("case14", (14, 20, 5), ["gencost", "bus_name"]),
    ("case30", (30, 41, 6), ["gencost"]),
    ("case9_features", (9, 10, 5), []),
])
def test_every_standard_case_reads(case, sizes, ignored):
    net = read_matpower(power_case(case))
    assert (len(net.buses), len(net.branches), len(net.generators)) == sizes
    assert net.name == case and net.notes["ignored_fields"] == ignored
    assert sum(b.kind == "slack" for b in net.buses) == 1


def test_a_byte_order_mark_is_read_and_undecodable_bytes_are_refused(tmp_path):
    p = _case(tmp_path, "")
    p.write_bytes(p.read_text(encoding="utf-8").encode("utf-8-sig"))
    assert read_matpower(p).name == "tiny"
    p.write_bytes(p.read_bytes().replace(b"function", b"\xff\xfe function"))
    with pytest.raises(ValueError, match=r"tiny\.m.*not UTF-8") as failure:
        read_matpower(p)
    assert isinstance(failure.value.__cause__, UnicodeDecodeError)


def test_rows_of_different_lengths_are_refused(tmp_path):
    p = _case(tmp_path, "")
    # the second bus row gets a 14th column; the first keeps 13
    p.write_text(p.read_text().replace(" 2 1 50 10 0 0 1 1 0 1 1 1.1 0.9;",
                                       " 2 1 50 10 0 0 1 1 0 1 1 1.1 0.9 7;"))
    with pytest.raises(ValueError, match=r"line 6: mpc.bus row has 14 columns.*13"):
        read_matpower(p)


def test_block_comments_are_refused(tmp_path):
    p = _case(tmp_path, "%{\nmpc.gencost = [1];\n%}\n")
    with pytest.raises(ValueError, match=r"line 14: block comments %\{ \.\.\. %\} are not"):
        read_matpower(p)
