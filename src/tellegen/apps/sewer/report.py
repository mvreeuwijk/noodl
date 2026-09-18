"""Reporting helpers: mg/L, ppm by volume, and a per-pipe results table."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from tellegen.apps.sewer.quality import ppm_from_concentration

Tensor = torch.Tensor
F64 = torch.float64

_COLUMNS = ("pipe", "q", "h", "v", "R_h", "d_m", "T", "A_air", "D_h", "V_wet", "V_air")


def to_mg_per_litre(x: Tensor) -> Tensor:
    """kg/m3 -> mg/L (the factor is exactly 1e3)."""
    return x * 1e3


def to_ppm(concentration: Tensor, temperature: Tensor) -> Tensor:
    """kg/m3 of headspace H2S -> ppm by volume at the headspace temperature (K)."""
    return ppm_from_concentration(concentration, temperature)


def pipe_table(model, state, drivers, *, path=None) -> list[dict]:
    """One row per pipe of every quantity `SewerHydraulics` publishes; optionally to CSV."""
    resolved = model._apply_closures(state, drivers)
    rows: list[dict] = []
    for i, name in enumerate(model.pipe_names):
        row = {"pipe": name}
        for column in _COLUMNS[1:]:
            row[column] = float(resolved[f"sewer.{column}"][..., i])
        rows.append(row)
    if path is not None:
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    return rows
