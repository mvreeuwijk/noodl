"""Reporting helpers."""

import csv

import pytest
import torch

from noodl.apps.sewer.network import build_sewer_model, sewer_steady, tree_steady
from noodl.apps.sewer.report import pipe_table, to_mg_per_litre, to_ppm

F64 = torch.float64


def test_mg_per_litre_is_a_thousandfold():
    assert float(to_mg_per_litre(torch.tensor(2.0e-3, dtype=F64))) == pytest.approx(2.0)


def test_ppm_matches_the_measured_equilibrium():
    value = to_ppm(torch.tensor(3.867789040e-4, dtype=F64),
                   torch.tensor(293.15, dtype=F64))
    assert float(value) == pytest.approx(272.990, abs=1e-3)


def test_pipe_table_writes_every_pipe(tmp_path):
    model, state, drivers = build_sewer_model(tree_steady())
    final = sewer_steady(model, state, drivers)
    path = tmp_path / "pipes.csv"
    rows = pipe_table(model, final, drivers, path=path)
    assert [row["pipe"] for row in rows] == ["C1", "C2", "C4", "C3", "C5"]
    assert rows[0]["q"] == pytest.approx(0.05, abs=1e-12)
    assert rows[0]["h"] == pytest.approx(0.153007001, abs=1e-9)
    with path.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert set(written[0]) >= {"pipe", "q", "h", "v", "A_air", "D_h"}
