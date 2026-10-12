"""CONTAM ambient-concentration file (.ctm) reader: TN 1887 section 3.9.7.

RECORD LAYOUT (verified on NIST's `contamxpy` 0.0.9 samples):

    line 1      `SpeciesFile ContamW 2.0`, the signature (anything else is refused)
    line 2      a free-text description (ignored)
    line 3      `<start date> <end date> <number of species>` (`M/D`)
    next N      one species name per line (CONTAM's names, matched to the project's)
    header      `!Date Time <names>`, then rows `M/D H:MM:SS x_1 .. x_N` (kg/kg) to the end;
                hours may be written with one digit (` 0:00:00`), and a day's `24:00:00` and
                the next day's `00:00:00` are the same instant
ContamX interpolates linearly between rows, and so does this reader. ContamX uses the file
only when its species match the project's; this reader refuses a mismatch by name instead of
ignoring the file silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from noodl.apps.building_physics.inputs import Input
from noodl.apps.building_physics.schedules import (
    DAY,
    Calendar,
    _file_breakpoints,
    _piecewise,
    _seconds_into_file,
)
from noodl.apps.building_physics.wth import _day_of_year, _seconds, _strip_comment

F64 = torch.float64


@dataclass
class AmbientConcentrations:
    """Ambient (outdoor) mass fractions over time: `t` (s from 00:00 of `start_day`, a day
    of the year), `x` `(len(t), K)` (kg/kg) for the `species` named, in file order."""

    t: torch.Tensor
    x: torch.Tensor
    species: tuple[str, ...]
    start_day: int = 1

    def signal(self, species: Sequence[str]) -> _AmbientSignal:
        """The concentrations of `species` (in that order) as a `Signal`; the file must carry
        exactly those species."""
        names = list(species)
        if sorted(names) != sorted(self.species):
            raise ValueError(
                f"ambient concentrations: the file carries {list(self.species)} but the "
                f"model's species are {names}; ContamX uses a contaminant file only when the "
                f"two match"
            )
        cols = torch.tensor([self.species.index(n) for n in names], dtype=torch.long)
        return _AmbientSignal(self, cols)

    def inputs(self, model, *, layer: str = "species") -> list[Input]:
        """One input writing the ambient mass fractions into `"<layer>.x_boundary"`."""
        names = model.transport[layer].species_names
        return [Input(self.signal(list(names)), f"{layer}.x_boundary", name="ambient file")]


class _AmbientSignal:
    """The ambient file's columns as a `Signal`: linear between rows, never extrapolated."""

    def __init__(self, ambient: AmbientConcentrations, cols: torch.Tensor) -> None:
        self.ambient, self._t = ambient, ambient.t.tolist()
        self._x = ambient.x[:, cols]

    def __repr__(self) -> str:
        return "ambient concentrations"

    def value(self, t: float, calendar: Calendar) -> torch.Tensor:
        s = _seconds_into_file(
            "ambient concentrations", t, calendar, self.ambient.start_day, self._t
        )
        return _piecewise(self._t, self._x, s, linear=True, left=False)

    def breakpoints(self, lo: float, hi: float, calendar: Calendar) -> list[float]:
        return _file_breakpoints(self._t, self.ambient.start_day, lo, hi, calendar)


def read_ctm(path) -> AmbientConcentrations:
    """A CONTAM `.ctm` ambient-concentration file as `AmbientConcentrations`."""
    lines = [_strip_comment(ln) for ln in Path(path).read_text().splitlines()]
    if not lines or not lines[0].startswith("SpeciesFile ContamW"):
        raise ValueError(
            f"ctm: {path} is not a CONTAM contaminant file (expected 'SpeciesFile ContamW 2.0')"
        )
    if len(lines) < 3 or len(lines[2].split()) < 3:
        raise ValueError(f"ctm: {path}: line 3 must be '<start> <end> <number of species>'")
    start, _end, n = lines[2].split()[:3]
    day0, K = _day_of_year(start), int(n)
    rest = [ln for ln in lines[3:] if ln]
    species = tuple(rest[:K])
    rows: list[tuple[float, list[float]]] = []
    for ln in rest[K:]:
        if ln.startswith("!"):
            continue
        tok = ln.split()
        if len(tok) != 2 + K:
            raise ValueError(f"ctm: {path}: row {ln!r} has {len(tok)} fields, expected {2 + K}")
        t = (_day_of_year(tok[0]) - day0) * DAY + _seconds(tok[1])
        if rows and t < rows[-1][0]:
            raise ValueError(f"ctm: {path}: time axis decreases at {tok[0]} {tok[1]}")
        rows.append((t, [float(v) for v in tok[2:]]))
    if len(rows) < 2:
        raise ValueError(f"ctm: {path} has fewer than two rows")
    return AmbientConcentrations(
        t=torch.tensor([r[0] for r in rows], dtype=F64),
        x=torch.tensor([r[1] for r in rows], dtype=F64),
        species=species, start_day=day0)
