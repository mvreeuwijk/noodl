"""Code-to-code comparison of noodl's circuits with OpenModelica on MSL and MBL examples.

The reference models are `Modelica.Electrical.Analog` (MSL 4.1.0) and `Buildings.Electrical`
(MBL v13.0.0, commit 55abf579) examples simulated by OpenModelica 1.27.1 at a DASSL tolerance
of 1e-12 (`tests/data/modelica_electrical/NOTICE.md`). Each model is read with
`read_modelica_circuit`, run with `simulate_circuit` on the reference CSV's own time grid, and
every compared column of the CSV is checked at EVERY row under `-m slow` (the default run
checks a subset of the rows, below).

Algebraic models (no capacitor, no inductor): the circuit is algebraic in its sources and
signals, so noodl's steady solve at each output time is the problem the DAE solver solves, and
agreement is to round-off:

    |noodl - omc| <= 1e-9 |omc| + 1e-12   (units of the column: V, A or K).

The tolerance is not loosened for any model, column or row. The steady solves use the
runner's own Newton tolerances (`run.STEADY_ATOL`, `run.STEADY_RTOL`; `THERMAL_STEADY_ATOL`
for a circuit with a heat-port thermal network); nothing looser is passed. MSL's `Resistor`
(a coupled thermal solve at every row) is compared at every row only under `-m slow`; the
default run compares every tenth row at the same bound (`SUBSAMPLED`).

Dynamic models: two fixed-step runs at `r` and `2 r` substeps, combined by Richardson
extrapolation, agree at every row after the start row to

    |noodl - omc| <= 1e-6 max|omc column|   (`PARITY_RTOL`, of the column's range),

with `r` per model in `PARITY_SUBSTEPS` (measured values in the comment above it). The
default run compares each model's first rows (`PARITY_ROWS`); `-m slow` every row.

MBL's `ACLine_RL` example is compared with its dynamic branch (`RL_3`, a `FixedZ_dynamic`
line, and its load `load_sc_3`) omitted; the source `E` feeds that branch too, so its current
columns are not this model's to compare and are skipped for that model only. Every other CSV
column must map to a noodl variable (`CircuitNames.value`).

The worst absolute and relative error per model and column is written, when the environment
variable `NOODL_RECORD_PARITY=1` is set, to the committed
`tests/data/modelica_electrical/parity-electrical.json` for the documentation. Recording is
opt-in: a plain checkout or a CI run never rewrites that file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.electrical.modelica import extrapolate, read_modelica_circuit
from noodl.apps.electrical.modelica.run import simulate_circuit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "modelica_electrical"
RECORD = DATA / "parity-electrical.json"

ALGEBRAIC = ("Analog.Examples.Resistor", "Analog.Examples.ShowVariableResistor",
             "DC.Loads.Examples.Resistor", "DC.Loads.Examples.VariableLoad",
             "AC.OnePhase.Loads.Examples.TestImpedance", "AC.OnePhase.Lines.Examples.ACLine_RL",
             "AC.OnePhase.Conversion.Examples.ACDCConverter",
             "AC.OnePhase.Conversion.Examples.ACACTransformer",
             "AC.ThreePhasesBalanced.Lines.Examples.ACLine")
DYNAMIC = ("Analog.Examples.SeriesResonance", "Analog.Examples.CauerLowPassAnalog",
           "Analog.Examples.CompareTransformers", "Analog.Examples.HeatingRectifier",
           "DC.Lines.Examples.RCModel")
OMIT = {"AC.OnePhase.Lines.Examples.ACLine_RL": ("RL_3", "load_sc_3")}
# E feeds the omitted RL_3 as well, so its current is not this model's to compare.
SKIPPED_COLUMNS = {"AC.OnePhase.Lines.Examples.ACLine_RL": ("E.terminal.i[1]",
                                                            "E.terminal.i[2]")}
RTOL_ALGEBRAIC, ATOL_ALGEBRAIC = 1e-9, 1e-12  # never loosened


def _csv(stem: str) -> tuple[list[str], np.ndarray]:
    lines = [ln for ln in (DATA / f"{stem}.csv").read_text().splitlines()
             if ln and not ln.startswith("#")]
    head = [h.strip('"') for h in lines[0].split(",")]
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
    return head, data


def _record(stem: str, stats: dict, record: Path = RECORD) -> None:
    # Deliberate side effect: the measured errors go into the committed
    # tests/data/modelica_electrical/parity-electrical.json for the documentation, only
    # when NOODL_RECORD_PARITY=1 is set -- a plain checkout or a CI run never rewrites it.
    if os.environ.get("NOODL_RECORD_PARITY") != "1":
        return
    lock = record.with_suffix(".lock")
    for _ in range(6000):  # one writer at a time: models may be recorded in parallel
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        raise TimeoutError(f"{lock} held for 5 minutes")
    try:
        try:
            doc = json.loads(record.read_text()) if record.exists() else {}
        except json.JSONDecodeError:
            doc = {}
        doc[stem] = stats
        record.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    finally:
        os.close(fd)
        os.remove(lock)


# MSL's Resistor example couples a heat-port thermal network (Joule heat in, temperature-
# dependent resistance back) at every row: ~95 ms a row, ~48 s over its 501 rows. Its full
# comparison is marked slow; the default run compares every tenth row. Every model listed
# here is slow in full and subsampled in the default run.
SUBSAMPLED = {"Analog.Examples.Resistor": 10}


def _check(stem: str, column: str, times: np.ndarray, mine: np.ndarray, omc: np.ndarray,
           *, scale: float | None = None, bound: float = 0.0) -> dict:
    """Assert `mine` is within the bound of `omc` at every row (a NaN or infinity on either
    side fails, naming the first such row). Without `scale`, the algebraic bound
    `RTOL_ALGEBRAIC |omc| + ATOL_ALGEBRAIC`, and the worst absolute and relative error; with
    it, the dynamic bound `bound * scale` (`scale` the column's range), and the worst
    absolute error and the worst error as a fraction of the range."""
    with np.errstate(invalid="ignore"):  # inf - inf: NaN, caught as off the bound below
        err = np.abs(mine - omc)
    if scale is None:
        within = err <= RTOL_ALGEBRAIC * np.abs(omc) + ATOL_ALGEBRAIC
    else:
        within = err <= bound * scale
    bad = ~(np.isfinite(mine) & np.isfinite(omc) & within)
    if bad.any():
        k = int(bad.argmax())
        raise AssertionError(f"{stem}: {column} off the bound from t = {float(times[k])!r}: "
                             f"noodl {float(mine[k])!r}, omc {float(omc[k])!r}")
    if scale is not None:
        return {"max_abs": float(err.max()), "max_rel_range": float(err.max() / scale)}
    return {"max_abs": float(err.max()),
            "max_rel": float((err / np.maximum(np.abs(omc), ATOL_ALGEBRAIC)).max())}


def _compare(stem: str, every: int = 1) -> dict:
    """Run `stem` at every `every`-th CSV row and check each compared column against the
    bound; the worst absolute and relative error per column."""
    head, data = _csv(stem)
    data = data[::every]
    model, state, drivers, names = read_modelica_circuit(DATA / f"{stem}.json",
                                                         omit=OMIT.get(stem, ()))
    assert not names.dynamic
    times = torch.as_tensor(data[:, 0])
    torch.testing.assert_close(names.times[::every], times, rtol=0, atol=1e-12)
    out = simulate_circuit(model, state, drivers, names,
                           times=None if every == 1 else names.times[::every])
    skipped = set(SKIPPED_COLUMNS.get(stem, ()))
    omitted = {h for h in head[1:] if h.split(".")[0] in OMIT.get(stem, ())}
    stats = {}
    for j, h in enumerate(head[1:], start=1):
        if h in skipped or h in omitted:
            continue
        assert h in names.columns, f"{stem}: CSV column {h!r} maps to no noodl variable"
        stats[h] = _check(stem, h, data[:, 0], names.value(out, h).numpy(), data[:, j])
    if names.phasor:  # DC parts of a phasor model stay real
        dc = [n for n in names.nodes if n.endswith(("[1]", "[2]"))]
        for n in dc:
            assert float(out["electrical.phi"][..., names.nodes[n]].imag.abs().max()) <= 1e-12
    return stats


@pytest.mark.parametrize("stem", [pytest.param(s, marks=pytest.mark.slow)
                                  if s in SUBSAMPLED else s for s in ALGEBRAIC])
def test_parity_on_the_algebraic_models(stem):
    _record(stem, _compare(stem))


@pytest.mark.parametrize("stem", sorted(SUBSAMPLED))
def test_parity_on_the_algebraic_models_at_subsampled_rows(stem):
    _compare(stem, every=SUBSAMPLED[stem])


# ------------------------------------------------------------------ dynamic models
# Each dynamic model is run at two step sizes, `substeps` r and 2r per interval of its step
# grid (PARITY_SUBSTEPS), and the two runs are combined by Richardson extrapolation
# (`extrapolate`, order 2: the steps are trapezoidal, backward Euler after a restart). Every
# compared column must agree with OpenModelica at every row after t0 (the start state is
# the reference's own, not compared) to
#
#     |noodl - omc| <= PARITY_RTOL * max|omc column|   (the column's range over every row),
#
# or to the bound of PARITY_BOUNDS, each documented with its demonstrated cause.
#
# Measured (15 Oct 2026): worst error of the extrapolation over every row after t0, as a
# fraction of the column's range, at the pair chosen and at the next finer pair, and the
# largest difference between the two extrapolations (the Richardson estimate is asymptotic
# where that difference is well below PARITY_RTOL and the error falls by 8 to 16 per
# halving: third order after the backward Euler restarts, fourth where the trapezoidal
# steps' even error expansion dominates). Each pair is the smallest r = 2^k that meets
# PARITY_RTOL with that margin.
#   RCModel              (2, 4)    1.7e-8;  (4, 8)    2.2e-9;  difference 1.5e-8
#   CauerLowPassAnalog   (8, 16)   1.2e-7;  (16, 32)  1.6e-8;  difference 1.1e-7
#                        C5.i at 1.1 s, just after the step at 1 s
#   SeriesResonance      (32, 64)  1.9e-7;  (64, 128) 1.2e-8;  difference 1.7e-7
#                        (16, 32) 3.0e-6: capacitor1.v at 0.63 s, past the resonance at
#                        0.5 s (100 Hz, Q = 20) of the 0 to 200 Hz sweep; (8, 16) 2.1e-6
#                        on the first 200 rows, where (16, 32) gives 1.8e-7
#   CompareTransformers  (16, 32)  3.1e-7;  (32, 64)  3.9e-8;  difference 2.7e-7
#                        resistor11.i at 0.1 s, the first period of the 10 Hz source
#   HeatingRectifier     (16, 32)  1.5e-7;  (32, 64)  1.7e-8;  difference 1.6e-7
#                        Capacitor1.i at 1.12 s; (8, 16) 1.1e-6 over the first 50 rows
# The default run compares each model's first rows (PARITY_ROWS): 9 to 17 s each on one
# worker of a lightly loaded machine (each step costs 10 to 30 ms), HeatingRectifier 33 s
# (its coupled steps about 100 ms; its window must reach the diode's first conduction, rows 5
# to 8). Each window holds the worst row of the model's first 50 to 200 rows (RCModel's first
# row, Cauer's step at 1 s, SeriesResonance's 2nd and 5th rows, CompareTransformers' 2nd,
# HeatingRectifier's 7th). SeriesResonance's default window ends at 7 ms, well before the
# resonance at 0.5 s, which only the full run reaches. `-m slow` compares every row, about 1
# to 42 minutes a model (measured 15 Oct 2026: RCModel 73 s, Cauer 306 s, HeatingRectifier
# 1,609 s, CompareTransformers 2,342 s, SeriesResonance 2,507 s).
PARITY_RTOL = 1e-6  # of each column's range; never loosened
PARITY_SUBSTEPS: dict[str, tuple[int, int]] = {
    "Analog.Examples.SeriesResonance": (32, 64), "Analog.Examples.CauerLowPassAnalog": (8, 16),
    "Analog.Examples.CompareTransformers": (16, 32),
    "Analog.Examples.HeatingRectifier": (16, 32), "DC.Lines.Examples.RCModel": (2, 4)}
PARITY_ROWS: dict[str, int] = {
    "Analog.Examples.SeriesResonance": 8, "Analog.Examples.CauerLowPassAnalog": 30,
    "Analog.Examples.CompareTransformers": 6, "Analog.Examples.HeatingRectifier": 8,
    "DC.Lines.Examples.RCModel": 100}
# Bounds above PARITY_RTOL, each with its demonstrated cause: none.
PARITY_BOUNDS: dict[str, dict[str, float]] = {}


def _dynamic(stem: str, rows: int | None, pair: tuple[int, int]) -> dict:
    """Run `stem` over its first `rows` CSV rows (every row: `None`) at `substeps` `pair`,
    extrapolate, and check every column after t0 against its bound; per column the worst
    error, absolute and as a fraction of the column's range, and the size of the
    extrapolation's correction (`own_error`, the finer run's own error, of the range)."""
    head, full = _csv(stem)
    data = full if rows is None else full[:rows]
    runs = []
    for r in pair:
        model, state, drivers, names = read_modelica_circuit(DATA / f"{stem}.json",
                                                             substeps=r)
        assert names.dynamic
        torch.testing.assert_close(names.times[:data.shape[0]], torch.as_tensor(data[:, 0]),
                                   rtol=0, atol=1e-12)
        runs.append(simulate_circuit(model, state, drivers, names,
                                     times=names.times[:data.shape[0]]))
    best = extrapolate(*runs, order=2)
    stats = {}
    for j, h in enumerate(head[1:], start=1):
        assert h in names.columns, f"{stem}: CSV column {h!r} maps to no noodl variable"
        scale = max(float(np.abs(full[:, j]).max()), ATOL_ALGEBRAIC)
        mine = names.value(best, h).numpy()[1:]
        fine = names.value(runs[1], h).numpy()[1:]
        s = _check(stem, h, data[1:, 0], mine, data[1:, j], scale=scale,
                   bound=PARITY_BOUNDS.get(stem, {}).get(h, PARITY_RTOL))
        s["own_error"] = float(np.abs(mine - fine).max() / scale)
        stats[h] = s
    return stats


@pytest.mark.parametrize("stem", DYNAMIC)
def test_dynamic_parity_on_the_first_rows(stem):
    _dynamic(stem, PARITY_ROWS[stem], PARITY_SUBSTEPS[stem])


@pytest.mark.slow
@pytest.mark.parametrize("stem", DYNAMIC)
def test_dynamic_parity_every_row(stem):
    _record(stem, _dynamic(stem, None, PARITY_SUBSTEPS[stem]))


def test_the_dynamic_bound_is_of_the_column_range_and_fails_on_a_nan():
    t = np.array([1.0, 2.0])
    omc = np.array([0.0, -4.0])
    s = _check("m", "c", t, omc + 3e-6, omc, scale=4.0, bound=1e-6)
    assert s["max_rel_range"] == pytest.approx(7.5e-7)
    with pytest.raises(AssertionError, match=r"m: c off the bound from t = 1\.0"):
        _check("m", "c", t, omc + 5e-6, omc, scale=4.0, bound=1e-6)
    with pytest.raises(AssertionError, match=r"t = 2\.0"):
        _check("m", "c", t, np.array([0.0, np.nan]), omc, scale=4.0, bound=1e-6)


def test_the_dynamic_tables_name_every_dynamic_model():
    assert set(PARITY_SUBSTEPS) == set(PARITY_ROWS) == set(DYNAMIC)
    assert set(PARITY_BOUNDS) <= set(DYNAMIC)
    assert all(b == 2 * a for a, b in PARITY_SUBSTEPS.values())


def test_the_bound_check_fails_on_a_nan_or_an_infinity():
    t = np.array([0.0, 1.0, 2.0])
    omc = np.array([1.0, 2.0, 3.0])
    assert _check("m", "c", t, omc.copy(), omc)["max_abs"] == 0.0
    for bad in (np.nan, np.inf):
        mine = omc.copy()
        mine[1] = bad
        with pytest.raises(AssertionError, match=r"m: c off the bound from t = 1\.0"):
            _check("m", "c", t, mine, omc)
        with pytest.raises(AssertionError, match=r"t = 1\.0"):
            _check("m", "c", t, omc, mine)


def test_every_fixture_is_parity_checked():
    stems = {p.stem for p in DATA.glob("*.json")} - {RECORD.stem}
    assert stems == set(ALGEBRAIC) | set(DYNAMIC)
