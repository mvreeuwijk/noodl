"""CONTAM ambient-concentration (.ctm) reader and its inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.ctm import read_ctm
from noodl.apps.building_physics.prj import project_to_model, read_prj
from noodl.apps.building_physics.schedules import Calendar

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
F64 = torch.float64


def test_reads_species_and_rows_across_two_days():
    a = read_ctm(DATA / "test_OneZoneWthCtmStack.ctm")
    assert a.species == ("sarin",) and a.start_day == 1
    assert a.x.shape == (a.t.numel(), 1)
    assert a.t[-1].item() == 2 * 86400.0
    s = a.signal(["sarin"])
    assert s.value(12 * 3600.0 + 150.0, Calendar()).tolist() == pytest.approx([0.0011627])
    assert s.value(86400.0 + 6 * 3600.0, Calendar()).tolist() == pytest.approx([0.0011627])


def test_signal_breakpoints_are_the_files_rows():
    s = read_ctm(DATA / "test_OneZoneWthCtmStack.ctm").signal(["sarin"])
    assert s.breakpoints(0.0, 13 * 3600.0, Calendar()) == [12 * 3600.0, 12 * 3600.0 + 300.0]


def test_ambient_file_species_must_match_the_model():
    a = read_ctm(DATA / "test_OneZoneWthCtmStack.ctm")
    with pytest.raises(ValueError, match=r"\['sarin'\].*\['CO2'\]"):
        a.signal(["CO2"])


def test_ambient_inputs_write_the_species_boundary():
    a = read_ctm(DATA / "test_OneZoneWthCtmStack.ctm")
    model, _, drivers = project_to_model(read_prj(DATA / "test_OneZoneWthCtmStack-UseApi.prj"))
    (inp,) = a.inputs(model)
    assert inp.key == "species.x_boundary" and inp.index is None
    v = inp.signal.value(300.0, Calendar())
    assert v.dtype == F64 and v.shape == (1,)


def test_a_signal_outside_the_file_raises():
    a = read_ctm(DATA / "valThreeZonesWthCtm.ctm")
    with pytest.raises(ValueError, match="outside"):
        a.signal(["sarin"]).value(2 * 86400.0, Calendar())


def test_a_non_ctm_file_is_refused(tmp_path):
    p = tmp_path / "x.ctm"
    p.write_text("WeatherFile ContamW 2.0\n")
    with pytest.raises(ValueError, match="SpeciesFile"):
        read_ctm(p)
