"""ContamX's exposure, age-of-air and air-change-rate result files, parsed without ContamX."""

from __future__ import annotations

from noodl.apps.building_physics.contamx import _result_files


def test_contamx_result_files_are_parsed(tmp_path):
    (tmp_path / "x.ebw").write_text(
        "2\t1\t1/1\t1/1\t300\t1\t0\nday\ttime\tctm\t1\t2\n"
        "1/1\t00:05:00\t1\t1.000000e-04\t2.000000e-05\n"
        "1/1\t00:10:00\t1\t3.000000e-04\t0.000000e+00\n")
    (tmp_path / "x.age").write_text(
        "3\t1/1\t1/1\t300\t1\t0\nday\ttime\t1\t2\t3\n"
        "1/1\t00:05:00\t1.0\t3.0\t6.0\n1/1\t00:10:00\t1.5\t3.5\t6.5\n")
    (tmp_path / "x.ach").write_text(
        "1/1\t1/1\t300\t1\t0\t900.00\nday\ttime\tpath\tduct\ttotal\n"
        "1/1\t00:05:00\t3.5e-01\t0.0\t3.5e-01\n1/1\t00:10:00\t3.6e-01\t0.0\t3.6e-01\n")
    out = _result_files(tmp_path / "x.prj")
    assert out["exposure"].shape == (2, 2, 1)
    assert out["exposure"][1, 0, 0].item() == 3e-4 and out["exposure"][1, 1, 0].item() == 0.0
    assert out["age"].tolist() == [[1.0, 3.0, 6.0], [1.5, 3.5, 6.5]]
    assert out["air_change_rate"].tolist() == [0.35, 0.36]
    assert out["building_volume"] == 900.0
    assert _result_files(tmp_path / "none.prj") == {}


def test_an_exposure_file_with_two_species_groups_each_steps_rows(tmp_path):
    """ContamX writes one exposure row per step and species, the step's species together
    (the `ctm` column numbers them), with CRLF line ends: the rows of a step become its
    (n_exposures, K) block."""
    rows = ["2\t2\t1/1\t1/1\t300\t1\t0", "day\ttime\tctm\t1\t2",
            "1/1\t00:05:00\t1\t1.0e-04\t2.0e-05", "1/1\t00:05:00\t2\t5.0e-01\t6.0e-01",
            "1/1\t00:10:00\t1\t3.0e-04\t4.0e-05", "1/1\t00:10:00\t2\t7.0e-01\t8.0e-01"]
    (tmp_path / "y.ebw").write_bytes(("\r\n".join(rows) + "\r\n").encode())
    out = _result_files(tmp_path / "y.prj")
    assert out["exposure"].tolist() == [[[1e-4, 0.5], [2e-5, 0.6]],
                                        [[3e-4, 0.7], [4e-5, 0.8]]]


def test_result_files_left_from_an_earlier_run_are_not_copied(tmp_path):
    """ContamX writes the result files only when the project's output flags ask for them, so
    a stale `.age` (or `.ebw`, `.ach`) beside the project would otherwise be read back as
    this run's."""
    from noodl.apps.building_physics.contamx import _isolated

    prj = tmp_path / "x.prj"
    prj.write_text("project\n")
    for suffix in (".ebw", ".age", ".ach", ".wth"):
        prj.with_suffix(suffix).write_text("stale\n")
    with _isolated(prj) as copy:
        assert sorted(p.name for p in copy.parent.iterdir()) == ["x.prj", "x.wth"]


def test_malformed_result_files_are_refused_by_name(tmp_path):
    """An empty file, a header without step rows, and an `.age` header whose zone count
    disagrees with its columns are refused with the file's name, not an IndexError."""
    import pytest

    cases = {
        "empty.age": "",
        "rows.ach": "1/1\t1/1\t300\t1\t0\t900.00\nday\ttime\tpath\tduct\ttotal\n",
        "zones.age": "3\t1/1\t1/1\t300\t1\t0\nday\ttime\t1\t2\n1/1\t00:05:00\t1.0\t3.0\n",
        "species.ebw": "2\t1\t1/1\t1/1\t300\t1\t0\nday\ttime\tctm\t1\t2\n"
                       "1/1\t00:05:00\t1\t1.0e-04\n",
    }
    for name, text in cases.items():
        path = tmp_path / name
        path.write_text(text)
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            _result_files(path.with_suffix(".prj"))
        path.unlink()
