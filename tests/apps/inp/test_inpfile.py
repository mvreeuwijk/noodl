"""The shared section-keyed .inp tokenizer both readers build on."""

import pytest

from noodl.apps.inpfile import InpLine, as_float, read_sections, require_fields

SAMPLE = """\
[TITLE]
;;A comment line
A title line

[OPTIONS]
FLOW_UNITS           CMS
FLOW_ROUTING         KINWAVE

[JUNCTIONS]
;;Name  Elev
J1      12.0   5    ; trailing comment
J2      11.0   5
[junctions]
J3      10.0   5
"""


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.inp"
    path.write_text(SAMPLE)
    return path


def test_sections_are_upper_cased_and_merged(sample):
    sections = read_sections(sample)
    assert set(sections) == {"TITLE", "OPTIONS", "JUNCTIONS"}
    assert [line.fields for line in sections["JUNCTIONS"]] == [
        ("J1", "12.0", "5"), ("J2", "11.0", "5"), ("J3", "10.0", "5")
    ]


def test_comments_and_blank_lines_are_dropped(sample):
    sections = read_sections(sample)
    assert [line.fields for line in sections["TITLE"]] == [("A", "title", "line")]


def test_line_numbers_are_the_files_own(sample):
    sections = read_sections(sample)
    assert [line.number for line in sections["JUNCTIONS"]] == [11, 12, 14]


def test_raw_text_is_preserved(sample):
    line = read_sections(sample)["JUNCTIONS"][0]
    assert line.raw.startswith("J1")


def test_content_before_any_section_is_refused(tmp_path):
    path = tmp_path / "bad.inp"
    path.write_text("J1 12.0\n[JUNCTIONS]\n")
    # FR-5: punctuation (a colon) right after the line number, matching every other
    # refusal in this module.
    with pytest.raises(ValueError, match="line 1: .* before any .section."):
        read_sections(path)


def test_a_malformed_header_is_refused(tmp_path):
    path = tmp_path / "bad.inp"
    path.write_text("[JUNCTIONS\nJ1 12.0\n")
    with pytest.raises(ValueError, match="line 1"):
        read_sections(path)


def test_a_header_with_more_than_one_bracket_pair_is_refused(tmp_path):
    """FR-4: `[JUNC[TIONS]` ends with `]` and has length >= 3, but is not a single
    `[NAME]` pair -- it must be refused as a malformed header, not silently read as the
    section name `JUNC[TIONS`."""
    path = tmp_path / "bad.inp"
    path.write_text("[JUNC[TIONS]\nJ1 12.0\n")
    with pytest.raises(ValueError, match=r"line 1: a section header must read"):
        read_sections(path)


def test_require_fields_names_the_line(sample):
    line = InpLine(section="JUNCTIONS", number=11, fields=("J1",), raw="J1")
    with pytest.raises(ValueError, match="line 11 of .*: a junction needs 2 fields, got 1"):
        require_fields(line, 2, "a junction", sample)


def test_as_float_names_the_field(sample):
    line = InpLine(section="JUNCTIONS", number=11, fields=("J1", "abc"), raw="J1 abc")
    with pytest.raises(ValueError, match="line 11 of .*: the invert must be a number"):
        as_float(line, 1, "the invert", sample)


def test_as_float_returns_the_value(sample):
    line = InpLine(section="JUNCTIONS", number=11, fields=("J1", "12.5"), raw="")
    assert as_float(line, 1, "the invert", sample) == 12.5
